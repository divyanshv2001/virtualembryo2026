"""Past-only transport transfer diagnostic on exposed challenge development data."""
import argparse
import json
import importlib.metadata
from collections import Counter
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now,append_event
from hurdle_backtest import read_cells
from challenge_transfer import ChallengeTransfer
from dimensional_transport_flow import DimensionalTransportFlow
from cnf_manifold_flow import DensityFlowNet,train_manifold_density
from feature_panel_forecast import FeaturePanelForecast
from log1p_positive_forecast import Log1pPositiveForecast
from latent_population_scale import LatentPopulationScale
from sklearn.decomposition import PCA
from ridge_conditional_head import RidgeConditionalPositiveForecast
from transport_anchor_alignment import predict_aligned
from offline_backtest import load_core,Panel


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    root=HERE.parents[1];data=HERE/'private/associated_prepared_01';out=HERE/'private/cnf_latent_population_scale_challenge_02'
    if out.exists() and any(out.iterdir()) and not args.resume:raise ValueError('Preserve prior run; use --resume')
    prepared=json.loads((data/'report.json').read_text())
    for f,k in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(data/f)!=prepared[k]:raise ValueError('Prepared data changed')
    anchor_path=root/'data/E8.5_RNA.h5ad';target_path=root/'data/E9.5_RNA.h5ad'
    panel=(root/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
    controls=['copy','anchorslope_both_0.5','log1p_abundance_a0.5','log1p_joint_a0.5']
    configs=controls+[f'population_scale_{mode}_{factor}' for mode in ['abundance','joint'] for factor in [.75,1.25]]
    old=HERE/'private/cnf_feature_challenge_01'
    encoder_root=HERE/'private/transport_challenge_01'
    plan={'created_before_training_utc':now(),'configs':configs,'evaluation_seeds':[20260928,20260929,20260930],
        'cutoff':8.5,'target':9.5,'donor_seed':20260928,'donor_count':1500,'truth_count':1000,'ceiling_count':1000,
        'mechanisms':['abundance','joint'],'population_factors':[.75,1.25],
        'source_sha256':{f:digest(HERE/f) for f in ['cnf_latent_population_scale_challenge.py','latent_population_scale.py','anchor_slope_calibration.py','log1p_positive_forecast.py','transport_anchor_alignment.py','hurdle_backtest.py','hurdle_transfer.py','detection_transfer.py','challenge_transfer.py','robust_population.py','constrained_forecast.py','transfer_genes.py','feature_panel_forecast.py','windowed_positive_head.py','cnf_manifold_flow.py','cnf_density_flow.py','dimensional_transport_flow.py','transport_latent_flow.py','ridge_conditional_head.py','neural_hurdle_forecast.py','offline_backtest.py']},
        'input_sha256':{p.name:digest(p) for p in [anchor_path,target_path]},'prepared_report_sha256':digest(data/'report.json'),
        'panel_sha256':digest(root/'outputs/t1_run/T1__val.genes.txt'),
        'dependencies':{k:importlib.metadata.version(k) for k in ['numpy','scipy','torch','scikit-learn','anndata']},
        'encoder_sha256':digest(old/'encoder4096.npz'),'heads_sha256':digest(old/'features4096_heads.npz'),
        'reference_checkpoint_sha256':digest(old/'features4096.pt'),
        'archived_report_sha256':digest(old/'report.json'),'archived_features_sha256':digest(old/'features.npy'),
        'hypothesis':'Latent population width may mismatch later cell heterogeneity. Contract/expand centered evolved coordinates with no latent mean drift; test full-expression distribution and covariance after decoding.',
        'fit':'Frozen4096gene8D800step field, past-only log1p positive ridge1 heads, positive/detection anchor slopes .5. Terminal centered scale factor**duration. Factor1 must replay controls exactly. Own sensitivity, no literature reproduction, no future-fit or inferred biological rate.',
        'scope':'Four population scale variants(.75/1.25 x abundance/joint) plusfourcontrols x3unchanged development panels=24 scores. Full32285gene scorer and originalcalibration. Source13963past cells and1500anchor sample, notfull430339atlas/fullchallenge. Reused E9.5, no independent embryos. Population centering uses all forecast donor rows; batch dependent and not an individual cell dynamical law.',
        'gate':'No official submission. Three panels cannot pass64replicate mean/lower-tail>72 or temporal gate.',
        'strength':1.,'training_steps':0,'abundance_factor_cap':2,'covariance_limit':.4,'submissions_allowed':0,'jev_requests_allowed':0}
    def control_path(name):return HERE/'private/cnf_log1p_positive_challenge_01'/(name+'.npy')
    plan['archived_control_sha256']={n:digest(control_path(n)) for n in controls}
    if args.resume:
        previous=json.loads((out/'plan.json').read_text())
        for k in [key for key in plan if key!='created_before_training_utc']:
            if previous[k]!=plan[k]:raise ValueError('Resume mismatch:'+k)
        plan=previous
    else:
        out.mkdir(exist_ok=True);(out/'plan.json').write_text(json.dumps(plan,indent=2))
        for f in plan['source_sha256']:(out/f).write_bytes((HERE/f).read_bytes())
    events=out/'events.jsonl';append_event(events,'challenge_transport_plan_resumed' if args.resume else 'challenge_transport_plan_frozen',sha256=digest(out/'plan.json'))
    x=np.load(data/'expression.npy',mmap_mode='r');stages=pd.read_csv(data/'selected_metadata.csv').numeric_stage.to_numpy(float)
    symbols=pd.read_csv(data/'genes.csv').symbol.fillna('').tolist()
    donors,donor_rows=read_cells(anchor_path,panel,1500,plan['donor_seed']);np.save(out/'anchor_rows.npy',donor_rows)
    generation_path=out/'generation.json'
    if not generation_path.exists():
        features=np.load(old/'features.npy');np.save(out/'features.npy',features)
        np.testing.assert_array_equal(donor_rows,np.load(old/'anchor_rows.npy'))
        old_generation=json.loads((old/'generation.json').read_text());old_generation['joint_systematic_s1.0']=json.loads((HERE/'private/cnf_hurdle_challenge_01/generation.json').read_text())['joint_systematic_s1.0'];generation={}
        def save(name,pred,indices,audit):
            np.save(out/(name+'.npy'),pred);np.save(out/(name+'_indices.npy'),indices)
            generation[name]={'audit':audit,'prediction_sha256':digest(out/(name+'.npy'))}
            append_event(events,'challenge_transport_forecast_frozen',candidate=name,prediction_sha256=generation[name]['prediction_sha256'])
        for name in controls:
            source=control_path(name);pred=np.load(source);indices=np.load(source.with_name(name+'_indices.npy'))
            if digest(source)!=plan['archived_control_sha256'][name]:raise ValueError('Control changed')
            save(name,pred,indices,{**json.loads((source.parent/'generation.json').read_text())[name]['audit'],'replayed_from':str(source)})
        guard_features=features.copy()
        encoder=np.load(old/'encoder4096.npz');encoder_features=encoder['features']
        center=encoder['center'];scale=encoder['scale']
        net=DensityFlowNet(encoder['basis'],encoder['pca_center'],8.5,7.25)
        saved=torch.load(old/'features4096.pt',weights_only=False,map_location='cpu');net.load_state_dict(saved['net']);net.eval()
        program=Log1pPositiveForecast(x,stages,8.5,donors,panel,symbols,net,center,scale,encoder_features,guard_features)
        append_event(events,'past_only_log1p_response_heads_fitted',fit_max_stage=8.5,model_audit=program.audit)
        np.savez_compressed(out/'hurdle_heads.npz',detection=program.detection,pmean=program.pmean,positive_coef=program.positive_coef,positive_center=program.positive_center,positive_mean=program.positive_mean,zcenter=program.zcenter,support=program.support)
        np.savez_compressed(out/'anchor_offsets.npz',positive_delta=program.anchor_positive_delta,detection_delta=program.anchor_detection_delta,positive_center=program.anchor_positive_center,latent_mean=program.anchor_latent_mean,count=program.anchor_count,support=program.calibration_support)
        program.configure(.5,.5)
        spread=LatentPopulationScale(net,1.);program.net=spread
        for mode in plan['mechanisms']:
            np.testing.assert_array_equal(program.predict(9.5,mode,1.,sampling='systematic')[0],np.load(control_path(f'log1p_{mode}_a0.5')))
        append_event(events,'unit_population_scale_replays_both_archived_forecasts_exactly')
        for mode in plan['mechanisms']:
            for factor in plan['population_factors']:
                spread.configure(factor)
                name=f'population_scale_{mode}_{factor}'
                prediction,indices,audit=program.predict(9.5,mode,1.,sampling='systematic')
                audit.update(terminal_population_factor=factor,population_scale_policy='Centered forecast population scaling factor**duration; latent mean retained, full expression means/covariances still must be measured. Own sensitivity, not learned biological variance.')
                save(name,prediction,indices,audit)
        if set(generation)!=set(configs):raise ValueError('Forecast coverage mismatch')
        generation_path.write_text(json.dumps(generation,indent=2))
        append_event(events,'all_challenge_forecasts_frozen_before_target_read',candidates=configs)
    generation=json.loads(generation_path.read_text())
    for name in configs:
        if digest(out/(name+'.npy'))!=generation[name]['prediction_sha256']:raise ValueError('Frozen forecast changed')
    core,_=load_core();report={'plan':plan,'generation':generation,'panels':[],'official_score':None,'local_gate_passed':False,'scorer_manifest_sha256':digest(HERE/'private/scorer_source/manifest.json')}
    if args.resume and (out/'report.partial.json').exists():report=json.loads((out/'report.partial.json').read_text());report.pop('in_progress_panel',None)
    for seed in plan['evaluation_seeds']:
        if any(p['seed']==seed for p in report['panels']):continue
        future,rows=read_cells(target_path,panel,2000,seed);np.save(out/f'target_rows_{seed}.npy',rows)
        order=np.random.default_rng(seed).permutation(len(future));evaluator=Panel(core,future[order[:1000]],donors,seed)
        floor=evaluator.metrics(donors);ceiling=evaluator.metrics(future[order[1000:]])
        previous=next(p for p in json.loads((old/'report.json').read_text())['panels'] if p['seed']==seed)
        if floor!=previous['floor'] or ceiling!=previous['ceiling']:raise ValueError('Calibration panels changed')
        results=[]
        for name in configs:
            raw=evaluator.metrics(np.load(out/(name+'.npy'),mmap_mode='r'))
            result={'candidate':name,'prediction_sha256':generation[name]['prediction_sha256'],'raw_metrics':raw,**evaluator.aggregate(raw,floor,ceiling)}
            results.append(result);append_event(events,'challenge_transport_candidate_scored',seed=seed,**result)
            (out/'report.partial.json').write_text(json.dumps({**report,'in_progress_panel':{'seed':seed,'cutoff':8.5,'target':9.5,'floor':floor,'ceiling':ceiling,'results':results}},indent=2))
        report['panels'].append({'seed':seed,'cutoff':8.5,'target':9.5,'floor':floor,'ceiling':ceiling,'results':results})
        (out/'report.partial.json').write_text(json.dumps(report,indent=2))
    for path in [anchor_path,target_path]:
        if digest(path)!=plan['input_sha256'][path.name]:raise ValueError('Challenge input changed')
    report['status']='completed';(out/'report.json').write_text(json.dumps(report,indent=2))
    summaries=[]
    for name in configs:
        results=[next(r for r in p['results'] if r['candidate']==name) for p in report['panels']]
        valid=all(r['calibration_valid'] for r in results);scores=[r['local_score'] for r in results]
        summaries.append({'candidate':name,'scores':scores,'mean_score':float(np.mean(scores)) if valid else None,'all_calibrations_valid':valid,'raw_metrics':[r['raw_metrics'] for r in results],'skills':[r['skills'] for r in results]})
    public={'updated_utc':now(),'status':'completed_not_promoted','summaries':summaries,'panels':report['panels'],'plan_sha256':digest(out/'plan.json'),'report_sha256':digest(out/'report.json'),'scope':plan['scope'],'local_72_gate_passed':False,'official_score':None,'submissions_used':0}
    (HERE/'CNF_LATENT_POPULATION_SCALE_RESULTS.json').write_text(json.dumps(public,indent=2))
    p=HERE/'LOCAL_OPTIMIZATION_STATE.json';state=json.loads(p.read_text());state['cnf_latent_population_scale_job'].update(status='completed_not_promoted',evaluations=24,pending_evaluations=0,report_sha256=public['report_sha256'],results_report='CNF_LATENT_POPULATION_SCALE_RESULTS.json');state['local_process_running']=False;state['active_jobs']=[];p.write_text(json.dumps(state,indent=2))
    p=HERE/'METRIC_RESEARCH_QUEUE.json';queue=json.loads(p.read_text())
    for entry in queue['paths']:
        if entry.get('run')=='cnf_latent_population_scale_challenge_02':entry.update(status='implemented_evaluated_not_promoted',results_report='CNF_LATENT_POPULATION_SCALE_RESULTS.json')
    p.write_text(json.dumps(queue,indent=2))
    from index_scores import main as index_scores
    index_scores();append_event(events,'challenge_transport_completed',evaluations=24)
    from update_compact_checkpoint import refresh
    refresh('Completed24 mean-preserving latent population scale scores. All means/raw4/skills4 retained in CNF_LATENT_POPULATION_SCALE_RESULTS.json and SCORE_LEDGER.jsonl. Review matched gains before promotion;72 gate remains unmet.')


if __name__=='__main__':
    try:
        torch.set_num_threads(2)
        with threadpool_limits(limits=2):main()
    except Exception as exc:
        folder=HERE/'private/cnf_latent_population_scale_challenge_02'
        if folder.exists():(folder/'failure.json').write_text(json.dumps({'status':'execution_failed','type':type(exc).__name__,'error':str(exc),'resume_command':'outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/cnf_latent_population_scale_challenge.py --resume'},indent=2))
        raise
