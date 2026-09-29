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
from cdr_logistic_forecast import CDRLogisticForecast
from sklearn.decomposition import PCA
from ridge_conditional_head import RidgeConditionalPositiveForecast
from transport_anchor_alignment import predict_aligned
from offline_backtest import load_core,Panel


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    root=HERE.parents[1];data=HERE/'private/associated_prepared_01';out=HERE/'private/cnf_cdr_logistic_repair_02'
    if out.exists() and not args.resume:raise ValueError('Preserve prior run; use --resume')
    prepared=json.loads((data/'report.json').read_text())
    for f,k in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(data/f)!=prepared[k]:raise ValueError('Prepared data changed')
    anchor_path=root/'data/E8.5_RNA.h5ad';target_path=root/'data/E9.5_RNA.h5ad'
    panel=(root/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
    controls=['copy','unit16','features4096_s0.5','features4096_s1.0','joint_systematic_s1.0','logistic0.1_joint','logistic1.0_joint']
    configs=controls+[f'cdrlogistic{ridge}_{mode}' for ridge in [.1,1.] for mode in ['detection','joint']]
    old=HERE/'private/cnf_feature_challenge_01'
    encoder_root=HERE/'private/transport_challenge_01'
    plan={'created_before_training_utc':now(),'configs':configs,'evaluation_seeds':[20260928,20260929,20260930],
        'cutoff':8.5,'target':9.5,'donor_seed':20260928,'donor_count':1500,'truth_count':1000,'ceiling_count':1000,
        'source_sha256':{f:digest(HERE/f) for f in ['cnf_cdr_logistic_challenge.py','transport_anchor_alignment.py','hurdle_backtest.py','hurdle_transfer.py','detection_transfer.py','challenge_transfer.py','robust_population.py','constrained_forecast.py','transfer_genes.py','feature_panel_forecast.py','cdr_detection_forecast.py','cdr_logistic_forecast.py','logistic_detection_head.py','windowed_positive_head.py','cnf_manifold_flow.py','cnf_density_flow.py','dimensional_transport_flow.py','transport_latent_flow.py','ridge_conditional_head.py','neural_hurdle_forecast.py','offline_backtest.py']},
        'input_sha256':{p.name:digest(p) for p in [anchor_path,target_path]},'prepared_report_sha256':digest(data/'report.json'),
        'panel_sha256':digest(root/'outputs/t1_run/T1__val.genes.txt'),
        'dependencies':{k:importlib.metadata.version(k) for k in ['numpy','scipy','torch','scikit-learn','anndata']},
        'fit':'Same13963 permitted prepared atlas rows<=E8.5 and archived384 features. Archived8D past PCA, whitened using all permitted past coordinate variances. Gaussian base at7.25 before first prepared7.5 snapshot. Time-conditioned3x64 tanh field; fixedRK4.125 and one Rademacher trace vector per minibatch64,400 Adam.001 updates, gradient cap5. All-past full conditional positive ridge1 heads; abundance cap2/covariance.4/mapped mass/protected genes unchanged. No latent velocity cap or biological growth/velocity input.',
        'scope':'Eight candidates across three unchanged frozen challengeE9.5 development panels;24 evaluations. Observation noise0/.05/.1 on whitened latent training cells; separate fixed RNG preserves stage/trace sampling. Density10/energy.1,800 updates fixed. No independent embryo validation; E9.5 previously exposed. Missing/ambiguous atlas genes preserve actual challenge donor values. Complete32285-gene scorer and original calibration unchanged.',
        'hypothesis':'Past-manifold density regularization may discourage unsupported transition paths. Compare midpoint5-neighbor Euclidean hinge at.1 with weights1/10 versus0, energy.1 fixed,400 updates and strengths.5/1. References are all permitted past cells, never future. Own adaptation, not full TrajectoryNet reproduction.',
        'gate':'No official submission or E10.5 export. Three panels cannot satisfy>=64 Monte Carlo gate. Mean/lower-tail>72 and temporal consistency still required. Retain every raw metric, skill and invalid calibration.',
        'manifold_fit':'Five-neighbor Euclidean hinge.1 at half-quarter inverse midpoint for past stages after earliest. All past latent cells as reference; energy.1 fixed; density weights0/1/10. Zero density model must replay prior400-step training exactly.',
        'strength':.5,'latent_velocity_cap':6,'abundance_factor_cap':2,'covariance_limit':.4,'submissions_allowed':0,'jev_requests_allowed':0}
    plan['past_encoder_sha256']=digest(encoder_root/'neutral_d8_flow.npz')
    plan['author_reference_sha256']=digest(HERE/'CNF_AUTHOR_REFERENCE.json')
    plan['manifold_reference_sha256']=digest(HERE/'CNF_MANIFOLD_REFERENCE.json')
    plan['reference_checkpoint_sha256']=digest(old/'features4096.pt')
    plan['hypothesis']='Test observed CDR nuisance adjustment under Bernoulli logistic link with archived non-CDR logistic controls at matching L2 ridges. Separate CDR covariate contribution from detection-link and ridge changes.'
    plan['fit']='Frozen4096-gene8D800-update encoder/flow, all13963 permitted rows<=8.5, global positive head unchanged. Logistic design[h,cdr,intercept], observed CDR standardized on past rows, fixed80 updates, ridges.1/1 with unpenalized intercept. Same optimizer as archived logistic controls. Donor CDR constant through forecast.'
    plan['scope']='Four CDR-logistic detection/joint forecasts plus seven archives including matched non-CDR logisticjoint controls=33 full-panel development scores. No future fitting; constraints/calibration unchanged. Not MAST Bayesian/DE-inference reproduction.'
    plan['strength']=1.;plan['latent_velocity_cap']=None;plan['training_steps']=800;plan['integration_step']=.125
    plan['manifold_fit']='Archived density10/energy.1 unchanged; no field retraining.'
    plan['cdr_reference_sha256']=digest(HERE/'CDR_DETECTION_REFERENCE.json')
    plan['logistic_reference_sha256']=digest(HERE/'LOGISTIC_DETECTION_REFERENCE.json')
    plan['encoder_sha256']=digest(old/'encoder4096.npz');plan['heads_sha256']=digest(old/'features4096_heads.npz')
    previous_run=HERE/'private/cnf_cdr_logistic_repair_01'
    previous_plan=json.loads((previous_run/'plan.json').read_text())
    for key in ['input_sha256','prepared_report_sha256','panel_sha256','encoder_sha256','heads_sha256','reference_checkpoint_sha256']:
        if previous_plan[key]!=plan[key]:raise ValueError('Fitted-head reuse inputs changed:'+key)
    for module in ['cdr_logistic_forecast.py','cdr_detection_forecast.py','logistic_detection_head.py']:
        if digest(previous_run/module)!=previous_plan['source_sha256'][module]:raise ValueError('Original fitted module provenance changed')
    plan['reused_head_sha256']=digest(previous_run/'cdrlogistic0.1_heads.npz')
    plan['reused_head_original_plan_sha256']=digest(previous_run/'plan.json')
    plan['reused_head_scope']='First ridge.1 optimization completed before logging failure. Reuse saved fitted coefficients, original input hashes/normalization/module snapshots verified; no repeated ridge.1 training.'
    def control_path(name):
        return (HERE/'private/cnf_logistic_challenge_01' if name.startswith('logistic') else HERE/'private/cnf_hurdle_challenge_01' if name=='joint_systematic_s1.0' else old)/(name+'.npy')
    plan['archived_report_sha256']=digest(old/'report.json')
    plan['archived_features_sha256']=digest(old/'features.npy')
    plan['archived_control_sha256']={n:digest(control_path(n)) for n in controls}
    if args.resume:
        previous=json.loads((out/'plan.json').read_text())
        for k in ['configs','evaluation_seeds','source_sha256','input_sha256','prepared_report_sha256','panel_sha256','dependencies','archived_report_sha256','archived_features_sha256','archived_control_sha256','author_reference_sha256','past_encoder_sha256','manifold_reference_sha256','reference_checkpoint_sha256','encoder_sha256','heads_sha256','cdr_reference_sha256','logistic_reference_sha256','reused_head_sha256','reused_head_original_plan_sha256']:
            if previous[k]!=plan[k]:raise ValueError('Resume mismatch:'+k)
        plan=previous
    else:
        out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2))
        for f in plan['source_sha256']:(out/f).write_bytes((HERE/f).read_bytes())
    events=out/'events.jsonl';append_event(events,'challenge_transport_plan_resumed' if args.resume else 'challenge_transport_plan_frozen',sha256=digest(out/'plan.json'))
    x=np.load(data/'expression.npy',mmap_mode='r');stages=pd.read_csv(data/'selected_metadata.csv').numeric_stage.to_numpy(float)
    symbols=pd.read_csv(data/'genes.csv').symbol.fillna('').tolist()
    donors,donor_rows=read_cells(anchor_path,panel,1500,plan['donor_seed']);np.save(out/'anchor_rows.npy',donor_rows)
    generation_path=out/'generation.json'
    if not generation_path.exists():
        features=np.load(old/'features.npy');np.save(out/'features.npy',features)
        np.testing.assert_array_equal(donor_rows,np.load(old/'anchor_rows.npy'))
        old_generation=json.loads((old/'generation.json').read_text());old_generation['joint_systematic_s1.0']=json.loads((HERE/'private/cnf_hurdle_challenge_01/generation.json').read_text())['joint_systematic_s1.0'];old_generation.update({n:v for n,v in json.loads((HERE/'private/cnf_logistic_challenge_01/generation.json').read_text()).items() if n.startswith('logistic')});generation={}
        def save(name,pred,indices,audit):
            np.save(out/(name+'.npy'),pred);np.save(out/(name+'_indices.npy'),indices)
            generation[name]={'audit':audit,'prediction_sha256':digest(out/(name+'.npy'))}
            append_event(events,'challenge_transport_forecast_frozen',candidate=name,prediction_sha256=generation[name]['prediction_sha256'])
        for name in controls:
            source=control_path(name);pred=np.load(source);indices=np.load(source.with_name(name+'_indices.npy'))
            if digest(source)!=plan['archived_control_sha256'][name]:raise ValueError('Control changed')
            save(name,pred,indices,{**old_generation[name]['audit'],'replayed_from':str(source)})
        guard_features=features.copy()
        encoder=np.load(old/'encoder4096.npz');encoder_features=encoder['features']
        center=encoder['center'];scale=encoder['scale']
        net=DensityFlowNet(encoder['basis'],encoder['pca_center'],8.5,7.25)
        saved=torch.load(old/'features4096.pt',weights_only=False,map_location='cpu');net.load_state_dict(saved['net']);net.eval()
        program=FeaturePanelForecast(x,stages,8.5,donors,panel,symbols,net,center,scale,encoder_features,guard_features)
        heads=np.load(old/'features4096_heads.npz')
        for key in ['positive_coef','positive_center','positive_mean','zcenter','support']:np.testing.assert_array_equal(getattr(program,key),heads[key])
        for strength in [.5,1.]:
            replay=program.predict(9.5,'abundance',strength)[0]
            np.testing.assert_array_equal(replay,np.load(old/f'features4096_s{strength}.npy'))
        append_event(events,'archived_positive_heads_and_abundance_forecasts_verified_exactly')
        np.savez_compressed(out/'hurdle_heads.npz',detection=program.detection,pmean=program.pmean,positive_coef=program.positive_coef,positive_center=program.positive_center,positive_mean=program.positive_mean,zcenter=program.zcenter,support=program.support)
        for ridge in [.1,1.]:
            candidate=CDRLogisticForecast(x,stages,8.5,donors,panel,symbols,net,center,scale,encoder_features,guard_features,detection_ridge=ridge,saved_heads=(previous_run/'cdrlogistic0.1_heads.npz' if ridge==.1 else None),emit=lambda **kw:append_event(events,'cdr_logistic_block_fitted',ridge=ridge,**kw))
            np.testing.assert_array_equal(candidate.positive_coef,program.positive_coef)
            np.testing.assert_array_equal(candidate.predict(9.5,'abundance',1.)[0],np.load(old/'features4096_s1.0.npy'))
            np.savez_compressed(out/f'cdrlogistic{ridge}_heads.npz',coef=candidate.cdr_logistic_coef,cdr_mean=candidate.cdr_mean,cdr_scale=candidate.cdr_scale,donor_cdr=candidate.donor_cdr)
            append_event(events,'cdr_detection_global_abundance_verified_exactly',detection_ridge=ridge)
            for mode in ['detection','joint']:
                name=f'cdrlogistic{ridge}_{mode}'
                prediction,indices,audit=candidate.predict(9.5,mode,1.,sampling='systematic')
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
    (HERE/'CNF_CDR_LOGISTIC_RESULTS.json').write_text(json.dumps(public,indent=2))
    p=HERE/'LOCAL_OPTIMIZATION_STATE.json';state=json.loads(p.read_text());state['cnf_cdr_logistic_job'].update(status='completed_not_promoted',evaluations=33,pending_evaluations=0,report_sha256=public['report_sha256'],results_report='CNF_CDR_LOGISTIC_RESULTS.json');state['local_process_running']=False;state['active_jobs']=[];p.write_text(json.dumps(state,indent=2))
    p=HERE/'METRIC_RESEARCH_QUEUE.json';queue=json.loads(p.read_text())
    for entry in queue['paths']:
        if entry.get('run')=='cnf_cdr_logistic_repair_02':entry.update(status='implemented_evaluated_not_promoted',results_report='CNF_CDR_LOGISTIC_RESULTS.json')
    p.write_text(json.dumps(queue,indent=2))
    from index_scores import main as index_scores
    index_scores();append_event(events,'challenge_transport_completed',evaluations=33)


if __name__=='__main__':
    try:
        torch.set_num_threads(2)
        with threadpool_limits(limits=2):main()
    except Exception as exc:
        folder=HERE/'private/cnf_cdr_logistic_repair_02'
        if folder.exists():(folder/'failure.json').write_text(json.dumps({'status':'execution_failed','type':type(exc).__name__,'error':str(exc),'resume_command':'outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/cnf_cdr_logistic_challenge.py --resume'},indent=2))
        raise
