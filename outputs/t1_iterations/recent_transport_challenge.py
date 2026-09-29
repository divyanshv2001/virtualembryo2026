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
from recent_transport_flow import RecentTransportFlow
from ridge_conditional_head import RidgeConditionalPositiveForecast
from transport_anchor_alignment import predict_aligned
from offline_backtest import load_core,Panel


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    root=HERE.parents[1];data=HERE/'private/associated_prepared_01';out=HERE/'private/recent_transport_challenge_01'
    if out.exists() and not args.resume:raise ValueError('Preserve prior run; use --resume')
    prepared=json.loads((data/'report.json').read_text())
    for f,k in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(data/f)!=prepared[k]:raise ValueError('Prepared data changed')
    anchor_path=root/'data/E8.5_RNA.h5ad';target_path=root/'data/E9.5_RNA.h5ad'
    panel=(root/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
    controls=['copy','unit16','neutral_d16_r1_identity','neutral_d8_r1.0_identity','neutral_d8_r1.0_moments']
    configs=controls+[f'recent_d{d}_w{w}' for d in [8,16] for w in [1,2]]
    old=HERE/'private/transport_challenge_01'
    plan={'created_before_training_utc':now(),'configs':configs,'evaluation_seeds':[20260928,20260929,20260930],
        'cutoff':8.5,'target':9.5,'donor_seed':20260928,'donor_count':1500,'truth_count':1000,'ceiling_count':1000,
        'source_sha256':{f:digest(HERE/f) for f in ['recent_transport_challenge.py','transport_anchor_alignment.py','hurdle_backtest.py','hurdle_transfer.py','detection_transfer.py','challenge_transfer.py','robust_population.py','constrained_forecast.py','transfer_genes.py','recent_transport_flow.py','dimensional_transport_flow.py','transport_latent_flow.py','ridge_conditional_head.py','neural_hurdle_forecast.py','offline_backtest.py']},
        'input_sha256':{p.name:digest(p) for p in [anchor_path,target_path]},'prepared_report_sha256':digest(data/'report.json'),
        'panel_sha256':digest(root/'outputs/t1_run/T1__val.genes.txt'),
        'dependencies':{k:importlib.metadata.version(k) for k in ['numpy','scipy','torch','scikit-learn','anndata']},
        'fit':'Same13963 permitted prepared atlas rows <=E8.5, archived384 past-selected features,3000 past PCA cells,512 coupling cells/stage. Restrict only the affine velocity regression to last1/2 intervals. All past couplings are computed and retained, PCA and heads unchanged. Neutral has no growth prior; no donor alignment for new variants.',
        'scope':'Nine candidates across three unchanged frozen challengeE9.5 development panels;27 evaluations. No independent embryo validation; E9.5 previously exposed. Missing/ambiguous atlas genes preserve actual challenge donor values. Complete32285-gene scorer and original calibration unchanged.',
        'hypothesis':'Global historical transport drift may average obsolete intervals and push current cells toward earlier means. Fit affine velocities only on the latest one/two historical couplings with the same PCA, coupling sampling and all-past ridge1 heads. Test8D/16D identity forecasts. Own nonstationarity adaptation; no paper reproduction or future extrapolation guarantee.',
        'gate':'No official submission or E10.5 export. Three panels cannot satisfy>=64 Monte Carlo gate. Mean/lower-tail>72 and temporal consistency still required. Retain every raw metric, skill and invalid calibration.',
        'strength':.5,'latent_velocity_cap':6,'abundance_factor_cap':2,'covariance_limit':.4,'submissions_allowed':0,'jev_requests_allowed':0}
    plan['archived_report_sha256']=digest(old/'report.json')
    plan['archived_features_sha256']=digest(old/'features.npy')
    plan['archived_control_sha256']={n:digest(old/(n+'.npy')) for n in controls}
    if args.resume:
        previous=json.loads((out/'plan.json').read_text())
        for k in ['configs','evaluation_seeds','source_sha256','input_sha256','prepared_report_sha256','panel_sha256','dependencies','archived_report_sha256','archived_features_sha256','archived_control_sha256']:
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
        old_generation=json.loads((old/'generation.json').read_text());generation={}
        def save(name,pred,indices,audit):
            np.save(out/(name+'.npy'),pred);np.save(out/(name+'_indices.npy'),indices)
            generation[name]={'audit':audit,'prediction_sha256':digest(out/(name+'.npy'))}
            append_event(events,'challenge_transport_forecast_frozen',candidate=name,prediction_sha256=generation[name]['prediction_sha256'])
        for name in controls:
            pred=np.load(old/(name+'.npy'));indices=np.load(old/(name+'_indices.npy'))
            if digest(old/(name+'.npy'))!=plan['archived_control_sha256'][name]:raise ValueError('Control changed')
            save(name,pred,indices,{**old_generation[name]['audit'],'replayed_from':str(old/(name+'.npy'))})
        past=np.flatnonzero(stages<=8.5)
        for dim in [8,16]:
            baseline=RecentTransportFlow(x,stages,8.5,panel,symbols,features,past,np.zeros(len(past)),'neutral',cell_budget=512,pca_budget=3000,latent_dim=dim)
            archived=np.load(old/f'neutral_d{dim}_flow.npz')
            for field,value in [('coefficient',baseline.net.coefficient.numpy()),('basis',baseline.net.basis.numpy()),('center',baseline.center),('scale',baseline.scale)]:np.testing.assert_array_equal(value,archived[field])
            append_event(events,'global_velocity_reconstruction_bit_identical',latent_dimensions=dim)
            for window in [1,2]:
                flow=RecentTransportFlow(x,stages,8.5,panel,symbols,features,past,np.zeros(len(past)),'neutral',cell_budget=512,pca_budget=3000,latent_dim=dim,recent_intervals=window)
                name=f'recent_d{dim}_w{window}';flow.save(out/(name+'_flow.npz'))
                program=RidgeConditionalPositiveForecast(x,stages,8.5,donors,panel,symbols,flow.net,flow.center,flow.scale,features,ridge=1.)
                program.audit.update(latent_flow=flow.audit)
                # The velocity fit must not alter source encoder or gene heads.
                expected=np.load(old/f'd{dim}_r1.0_heads.npz')
                for field in ['detection','positive_coef','positive_center','positive_mean','pmean','zcenter','support']:np.testing.assert_array_equal(getattr(program,field),expected[field])
                pred,indices,audit=program.predict(9.5,'abundance',.5);save(name,pred,indices,audit)
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
    (HERE/'RECENT_TRANSPORT_CHALLENGE_RESULTS.json').write_text(json.dumps(public,indent=2))
    p=HERE/'LOCAL_OPTIMIZATION_STATE.json';state=json.loads(p.read_text());state['recent_transport_challenge_job'].update(status='completed_not_promoted',evaluations=27,pending_evaluations=0,report_sha256=public['report_sha256'],results_report='RECENT_TRANSPORT_CHALLENGE_RESULTS.json');state['local_process_running']=False;state['active_jobs']=[];p.write_text(json.dumps(state,indent=2))
    p=HERE/'METRIC_RESEARCH_QUEUE.json';queue=json.loads(p.read_text())
    for entry in queue['paths']:
        if entry.get('run')=='recent_transport_challenge_01':entry.update(status='implemented_evaluated_not_promoted',results_report='RECENT_TRANSPORT_CHALLENGE_RESULTS.json')
    p.write_text(json.dumps(queue,indent=2))
    from index_scores import main as index_scores
    index_scores();append_event(events,'challenge_transport_completed',evaluations=27)


if __name__=='__main__':
    try:
        torch.set_num_threads(2)
        with threadpool_limits(limits=2):main()
    except Exception as exc:
        folder=HERE/'private/recent_transport_challenge_01'
        if folder.exists():(folder/'failure.json').write_text(json.dumps({'status':'execution_failed','type':type(exc).__name__,'error':str(exc),'resume_command':'outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/recent_transport_challenge.py --resume'},indent=2))
        raise
