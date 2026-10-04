"""Fresh cutoff, matched scFM gradient-projection experiment; no official upload."""
import copy
import argparse
import json
import traceback
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from fresh_fullpanel_context import prepare, incremental_context
from frozen_fullpanel_scoring import score_frozen_forecasts
from anchored_soft_ot import train_anchored
from cnf_manifold_flow import DensityFlowNet, train_manifold_density
from full_anchor_slope_forecast import FullAnchorSlopeForecast
from temporary_forecast_cache import TemporaryForecastCache
from offline_backtest import load_core
from scnode_resource_preflight import peak_memory
from run_t1 import digest
from iterate import now, append_event

HERE=Path(__file__).resolve().parent
RUN=None
PUBLIC=None


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    global RUN,PUBLIC
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',default='scfm_projected_past_fold');args=parser.parse_args()
    entry=json.loads((HERE/'RESEARCH_HARNESS_MANIFEST.json').read_text())['experiments'][args.experiment]
    RUN=(HERE/entry['run']).resolve();PUBLIC=(HERE/entry['report']).resolve()
    if not RUN.is_relative_to(HERE/'private') or not PUBLIC.is_relative_to(HERE):raise ValueError('Evidence outside project')
    if RUN.exists() or PUBLIC.exists():raise ValueError('Never duplicate experiment')
    RUN.mkdir(parents=True);emit=lambda kind,**kw:append_event(RUN/'events.jsonl',kind,**kw)
    report={'status':'running','panels':[],'new_scoring_batch':True,'original_readiness_gate_passed':False,'official_score':None}
    cache=TemporaryForecastCache(HERE/'private/temporary_cache')
    try:
        spec=entry['protocol']
        prepared=HERE/'private/associated_prepared_01';prep=json.loads((prepared/'report.json').read_text())
        inputs={}
        for filename,key in [('expression.npy','expression_sha256'),('genes.csv','genes_sha256'),('selected_metadata.csv','metadata_sha256')]:
            inputs[filename]=digest(prepared/filename)
            if inputs[filename]!=prep[key]:raise ValueError('Prepared input changed')
        core,scorer=load_core()
        sources=['scfm_projected_past_fold.py','fresh_fullpanel_context.py','frozen_fullpanel_scoring.py','anchored_soft_ot.py','soft_ot_flow_matching.py','cnf_manifold_flow.py','cnf_density_flow.py','full_anchor_slope_forecast.py','log1p_positive_forecast.py','tigon_conditional_fullpanel.py','offline_backtest.py']
        if spec.get('learned_diffusion'): sources.append('learned_latent_diffusion.py')
        if spec.get('frozen_kinetic_run'): sources.append('graph_kinetic_residual.py')
        if spec.get('difference_attention'): sources.append('past_difference_attention.py')
        graph_prior_inputs={}
        if spec.get('graph_kinetics'):
            sources.append('graph_kinetic_residual.py')
            for name in ['adult_mouse_base_grn.parquet','past4096_adult_graph.npz','past4096_panel_features.npy','SOURCE_LICENSE.txt']:
                graph_prior_inputs[name]=digest(HERE/'private/flecs_adult_graph_01'/name)
        plan={'created_utc':now(),'protocol':spec,'source_sha256':{name:digest(HERE/name) for name in sources},'input_sha256':inputs,'scoring_seeds':spec['scoring_seeds'],'scorer_manifest':scorer,'submissions_allowed':0}
        if graph_prior_inputs:plan['graph_prior_sha256']=graph_prior_inputs
        save(RUN/'plan.json',plan);emit('plan_frozen',sha256=digest(RUN/'plan.json'))
        c=prepare(spec,RUN,emit);cutoff,target=spec['cutoff'],spec['target']
        if any(digest(HERE/name)!=expected for name,expected in plan['source_sha256'].items()):raise ValueError('Source changed after plan freeze')
        if len(np.flatnonzero(c['stages']==target))<2000:raise ValueError('Insufficient target metadata support')
        torch.manual_seed(spec['seed'])
        initial=DensityFlowNet(c['basis'],c['pca'].mean_,cutoff,float(c['stages'][c['past']].min())-.25)
        reuse=spec.get('reuse_run')
        if reuse:
            previous=HERE/reuse
            with np.load(previous/'fresh_encoder.npz') as old,np.load(RUN/'fresh_encoder.npz') as fresh:
                for key in old.files:np.testing.assert_array_equal(old[key],fresh[key])
            previous_plan=json.loads((previous/'plan.json').read_text())
            if previous_plan['input_sha256']!=inputs:raise ValueError('Reused input mismatch')
            saved=torch.load(previous/'cnf800.pt',weights_only=False,map_location='cpu')
            initial.load_state_dict(saved['net']);history=saved['history']
            if saved['steps']!=800 or saved['batch_size']!=64 or saved['density_weight']!=10.:raise ValueError('Reference metadata mismatch')
            initial.eval();baseline=copy.deepcopy(initial)
            saved=torch.load(previous/'projected_ot400.pt',weights_only=False,map_location='cpu')
            if saved['fit_max_stage']!=cutoff or saved['steps']!=400 or saved['batch_size']!=64:raise ValueError('Past projected checkpoint mismatch')
            baseline.load_state_dict(saved['net']);baseline.eval()
            plan['reused_checkpoint_sha256']={name:digest(previous/(name+'.pt')) for name in ['cnf800','projected_ot400']}
            save(RUN/'plan.json',plan);emit('reused_encoder_and_checkpoints_verified',sha256=digest(RUN/'plan.json'))
            flows={'cnf800':initial,'projected_ot400':baseline}
            if spec.get('frozen_kinetic_run'):
                from graph_kinetic_residual import KineticFlow
                frozen=HERE/spec['frozen_kinetic_run']
                frozen_plan=json.loads((frozen/'plan.json').read_text())
                if frozen_plan['input_sha256']!=inputs or frozen_plan['protocol']['cutoff']!=cutoff:raise ValueError('Frozen kinetic inputs changed')
                if digest(HERE/'graph_kinetic_residual.py')!=frozen_plan['source_sha256']['graph_kinetic_residual.py']:raise ValueError('Frozen kinetic mechanism changed')
                flows={'cnf800':initial}
                for name,expected in spec['frozen_checkpoint_sha256'].items():
                    path=frozen/(name+'.pt')
                    if digest(path)!=expected:raise ValueError('Frozen checkpoint hash mismatch')
                    saved=torch.load(path,weights_only=False,map_location='cpu')
                    if saved['steps']!=400 or saved['batch_size']!=64 or saved['fit_max_stage']!=cutoff or not saved['frozen_drift_exact']:raise ValueError('Frozen kinetic metadata mismatch')
                    state=saved['net'];edges=state['edges'].numpy()
                    model=KineticFlow(initial,state['gene_mean'].numpy(),edges[1],edges[0],saved['kind'])
                    model.load_state_dict(state)
                    for key,value in initial.state_dict().items():torch.testing.assert_close(model.drift.state_dict()[key],value,rtol=0,atol=0)
                    flows[name]=model.eval()
                report['fixed_candidate_validation']={'training_steps':0,'reward_eligible':False,'past_decoder_reconstruction':'Same deterministic past-only procedure; prior full-panel predictions must replay bit-exact before new horizon.'}
                emit('frozen_kinetic_checkpoints_verified',checkpoint_sha256=spec['frozen_checkpoint_sha256'])
                if spec.get('difference_attention'):
                    from past_difference_attention import PastAttentionFlow,train_attention
                    times=np.unique(c['stages'][c['past']])
                    centroids=np.stack([c['coordinates'][c['stages'][c['past']]==t].mean(0) for t in times])
                    report.pop('fixed_candidate_validation')
                    report['attention_scope']='Own past-centroid level versus interval-normalized difference attention. Same4776parameters/frozenkinetics/CUDA400step budget; no individual-cell histories or author reproduction.'
                    for name,kind in [('attention_level400','level'),('attention_difference400','difference')]:
                        torch.manual_seed(20261004)
                        model=PastAttentionFlow(flows['kinetic_none400'],times,centroids,kind)
                        train_attention(model,c['coordinates'],c['stages'][c['past']],RUN/(name+'.pt'),emit,steps=spec['steps'],batch=spec['batch_size'])
                        flows[name]=model;emit('candidate_training_finished',candidate=name,device='cuda:0')
            elif spec.get('graph_kinetics'):
                from graph_kinetic_residual import prepare_kinetic_models, train_kinetics
                kinetic_models,graph_audit=prepare_kinetic_models(initial,c,RUN,emit)
                report['kinetic_graph_audit']=graph_audit
                flows={'cnf800':initial}
                for name,model in kinetic_models.items():
                    train_kinetics(model,c['coordinates'],c['stages'][c['past']],RUN/(name+'.pt'),emit,steps=spec['steps'],batch=spec['batch_size'])
                    flows[name]=model.eval();emit('candidate_training_finished',candidate=name)
                report['kinetic_scope']='FLeCS-inspired approximate reconstructed-gene kinetic residual in frozen PCA/CNF/decoder. Real vs degree-preserving shuffled graph is primary; no-edge has fewer parameters. Possible adult binding, not signed causal regulation or author reproduction. Past physical stages only.'
            elif spec.get('learned_diffusion'):
                from learned_latent_diffusion import DiffusionFlow, train_diffusion, smoke_test
                report['diffusion_preflight']=smoke_test(initial)
                emit('diffusion_preflight_passed', **report['diffusion_preflight'])
                flows={'cnf800':initial}
                for name,kind in [('constant_diffusion400','constant'),('state_diffusion400','state')]:
                    torch.manual_seed(spec['diffusion_seed'])
                    model=DiffusionFlow(initial,kind,seed=spec['diffusion_seed'])
                    train_diffusion(model,c['coordinates'],c['stages'][c['past']],RUN/(name+'.pt'),emit,steps=spec['steps'],batch=spec['batch_size'])
                    flows[name]=model.eval();emit('candidate_training_finished',candidate=name)
                report['diffusion_scope']='Frozen past CNF drift/heads; learned diagonal diffusion; scDiffEq-inspired CPU adaptation, not author reproduction. Constant vs state dependence isolates mechanism; original CNF uses different RK4 step. No future loss, no growth inference.'
            elif spec.get('fixed_candidate_validation'):
                report['fixed_candidate_validation']={'training_steps':0,'reward_eligible':False,'reason':'New horizon evaluation of already-accounted frozen candidate; no repeat model reward.'}
            elif spec.get('representation_comparison'):
                ci=incremental_context(c,RUN,emit,chunk=spec['ipca_chunk'])
                torch.manual_seed(spec['seed'])
                ipca=DensityFlowNet(ci['basis'],ci['pca'].mean_,cutoff,float(ci['stages'][ci['past']].min())-.25)
                train_manifold_density(ipca,ci['coordinates'],ci['stages'][ci['past']],.1,RUN/'ipca_cnf800.pt',emit,steps=800,density_weight=10.)
                flows['ipca_cnf800']=ipca.eval();model=copy.deepcopy(ipca)
                train_anchored(model,ci['coordinates'],ci['stages'][ci['past']],cutoff,RUN/'ipca_projected400.pt',emit,conflict_projection=True)
                flows['ipca_projected400']=model.eval()
                report['representation_diagnostics']={'same_fit_rows':True,'fit_rows':3000,'features':4096,'rank':8,'ipca_chunk':spec['ipca_chunk'],'past_fit_explained_variance_ratio_sum':{'pca':float(c['pca'].explained_variance_ratio_.sum()),'ipca':float(ci['pca'].explained_variance_ratio_.sum())},'interpretation':'Fit diagnostic only, not benchmark score or independent validation.'}
            else:
                for name,weight in [('projected_ot800',0.),('projected_pair800',1.)]:
                    model=copy.deepcopy(baseline)
                    train_anchored(model,c['coordinates'],c['stages'][c['past']],cutoff,RUN/(name+'.pt'),emit,conflict_projection=True,latent_variogram_weight=weight)
                    flows[name]=model.eval();emit('candidate_training_finished',candidate=name)
        else:
            history=train_manifold_density(initial,c['coordinates'],c['stages'][c['past']],.1,RUN/'cnf800.pt',emit,steps=800,density_weight=10.)
            initial.eval();flows={'cnf800':initial}
            for name in ['likelihood400','raw_ot400','projected_ot400']:
                model=copy.deepcopy(initial)
                if name=='likelihood400':
                    train_manifold_density(model,c['coordinates'],c['stages'][c['past']],.1,RUN/(name+'.pt'),emit,steps=400,density_weight=10.)
                else:
                    train_anchored(model,c['coordinates'],c['stages'][c['past']],cutoff,RUN/(name+'.pt'),emit,conflict_projection=name=='projected_ot400')
                flows[name]=model.eval();emit('candidate_training_finished',candidate=name)
        reference=FullAnchorSlopeForecast(c['x'],c['stages'],cutoff,c['donors'],c['panel'],c['symbols'],initial,c['center'],c['scale'],c['features'],c['guard'],anchor_path=c['anchor_path'])
        reference.configure(.25,.75)
        reference.audit.update(fit_max_stage=cutoff,method='Fresh source-domain PCA8 paired scFM',training_history=history)
        references={name:reference for name in flows}
        if spec.get('representation_comparison'):
            ipca_reference=FullAnchorSlopeForecast(ci['x'],ci['stages'],cutoff,ci['donors'],ci['panel'],ci['symbols'],flows['ipca_cnf800'],ci['center'],ci['scale'],ci['features'],ci['guard'],anchor_path=ci['anchor_path'])
            ipca_reference.configure(.25,.75)
            ipca_reference.audit.update(fit_max_stage=cutoff,method='Fresh source-domain IncrementalPCA8 paired scFM')
            for name in ['ipca_cnf800','ipca_projected400']:references[name]=ipca_reference
        generation={}
        if spec.get('frozen_kinetic_run'):
            previous_generation=json.loads((HERE/spec['frozen_kinetic_run']/'generation.json').read_text())
            for name in spec.get('prior_candidates',spec['candidates']):
                if name=='copy':prior=c['donors'].copy()
                else:
                    reference.net=flows[name]
                    prior,_,_=reference.predict(spec['prior_target'],'joint',1.,sampling='systematic')
                if cache.put('_prior_'+name,prior)!=previous_generation[name]['prediction_sha256']:raise ValueError('Original kinetic forecast reconstruction changed')
                del prior
            emit('prior_horizon_forecasts_reconstructed_bit_exact',target=spec['prior_target'])
        for name in spec['candidates']:
            if name=='copy':pred,ids,audit=c['donors'].copy(),np.arange(len(c['donors'])),{'method':'persistence'}
            else:
                reference=references[name];reference.net=flows[name];pred,ids,audit=reference.predict(target,'joint',1.,sampling='systematic')
            np.testing.assert_array_equal(pred[:,c['protected']],c['donors'][ids][:,c['protected']])
            if not np.isfinite(pred).all() or (pred<0).any():raise ValueError('Invalid forecast')
            sha=cache.put(name,pred);del pred
            if name=='copy':replay,replay_ids=c['donors'].copy(),np.arange(len(c['donors']))
            else:replay,replay_ids,_=reference.predict(target,'joint',1.,sampling='systematic')
            if cache.put('_replay_'+name,replay)!=sha:raise ValueError('Exact replay failed')
            np.testing.assert_array_equal(ids,replay_ids);del replay
            np.save(RUN/(name+'_indices.npy'),ids)
            generation[name]={'prediction_sha256':sha,'audit':copy.deepcopy(audit)}
            emit('forecast_frozen_exact_replay_passed',candidate=name,prediction_sha256=sha)
        save(RUN/'generation.json',generation);report['generation']=generation
        if any(digest(HERE/name)!=expected for name,expected in plan['source_sha256'].items()):raise ValueError('Source changed before target scoring')
        emit('all_forecasts_frozen_before_target_expression',target=target)
        score_frozen_forecasts(core,c['raw_x'],c['stages'],c['donors'],c['mapped'],c['columns'],c['panel'],cutoff,target,plan,spec['candidates'],generation,cache,RUN,report,emit,spec)
        report[spec.get('contrast_label','projected_minus_raw_ot_mean_skills')]=report.pop('growth_enabled_minus_disabled_mean_skills')
    except Exception as exc:
        (RUN/'traceback.txt').write_text(traceback.format_exc());report.update(status='failed',error=type(exc).__name__+': '+str(exc),failed_utc=now(),passing_candidates=[])
    finally:cache.close()
    report['resource_peak_process_working_set_bytes']=peak_memory()
    if peak_memory()>=16*1024**3:report.update(status='failed',error='16GiB resource gate failed',passing_candidates=[])
    save(RUN/'report.json',report);report['report_sha256']=digest(RUN/'report.json');save(PUBLIC,report)
    emit('batch_finished',status=report['status'],report_sha256=report['report_sha256'])
    print(json.dumps({'status':report['status'],'report':str(PUBLIC),'error':report.get('error')}))


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
