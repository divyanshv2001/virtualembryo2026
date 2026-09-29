"""Past-only stage-conditioned historical transport and positive abundance forecasts."""
import json
import argparse
import importlib.metadata
from collections import Counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import torch
from stage_transport_flow import StageTransportFlow
from conditional_positive_head import FullConditionalPositiveForecast
from train_extended_atlas import HERE
from offline_backtest import load_core, Panel
from run_t1 import digest
from iterate import now, append_event


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    root=HERE.parents[1];data=HERE/'private/associated_prepared_01'
    out=HERE/'private/stage_transport_horizon_01'
    if out.exists() and not args.resume:raise ValueError('Preserve previous run; use --resume')
    prepared=json.loads((data/'report.json').read_text())
    for f,k in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(data/f)!=prepared[k]:raise ValueError('Prepared data changed')
    plan={'created_before_training_utc':now(),'folds':[[8.,9.],[8.25,9.25]],
        'configs':['copy','unit16','saved384_d0.0','conditional_neutral_s0.5','conditional_growth1_s0.5']+[f'stage_{m}_r{r}_s{s}' for m in ['neutral','growth1'] for r in [1.,10.] for s in [.5,1.]],
        'donor_count':1500,'truth_count':1000,'ceiling_count':1000,'seed':20260928,
        'prepared_report_sha256':digest(data/'report.json'),
        'source_sha256':{f:digest(HERE/f) for f in ['stage_transport_horizon.py','detection_transfer.py',
            'challenge_transfer.py','robust_population.py','constrained_forecast.py','transfer_genes.py','offline_backtest.py','annotation_trend.py','neural_ode_forecast.py','neural_hurdle_forecast.py','transport_latent_flow.py','conditional_positive_head.py','stage_transport_flow.py']},
        'fit':'Historical transport velocities with PCA16/3000 fit cells and256 transport cells per stage. Add explicit interval-midpoint developmental stage to ridge1 affine field. Fit full conditional positive abundance heads on all permitted past cells.',
        'primary_objective':'Unchanged full-panel joint headline score. No metric regressions allowed for promotion; retain original >72 mean/lower-tail and temporal gate.',
        'ablation':'Neutral/growth1 unbalanced transport, stage coefficient ridge1/10, strengths.5/1. Controls replay stationary full-conditional abundance.5 and persistence. Future quarter-day solver uses actual evolving midpoint time; velocity cap6, factor cap2, covariance guard.4, protected genes/mapped mass unchanged. No detection switches.',
        'scope':'Exposed one-day source-cohort development folds, not fresh validation. Prepared sampled cohort, limited PCA/coupling subsets and missing official gene placeholders. Temporal acceleration is extrapolated from only2/3 historical interval groups. This is a statistical stage ablation, not TrajectoryNet/CNF reproduction.',
        'gate':'Diagnostic only; no official upload. Reject failed calibration and preserve every result. Positive training or development gains do not certify hidden E10.5.',
        'literature':'Primary TrajectoryNet PDF sections3.2-5.2 reviewed; time-dependent state fields motivate this ablation. Its interpolation experiments do not establish future extrapolation. No likelihood/energy/velocity-loss reproduction is claimed. See STAGE_DYNAMICS_REFERENCE.json.',
        'submissions_allowed':0,'jev_requests_allowed':0}
    old=HERE/'private/transport_conditional_horizon_01'
    feature_root=HERE/'private/neural_ode_horizon_01'
    plan['archived_control_report_sha256']=digest(old/'report.json')
    plan['archived_control_predictions_sha256']={f'cutoff_{c}/{n}.npy':digest(old/f'cutoff_{c}'/(n+'.npy')) for c,_ in plan['folds'] for n in ['copy','unit16','saved384_d0.0','conditional_neutral_s0.5','conditional_growth1_s0.5']}
    proxy_root=HERE/'private/growth_prior_audit_repair_01'
    plan['growth_proxy_sha256']={str(c):digest(proxy_root/f'scores_{c}.npz') for c,_ in plan['folds']}
    plan['feature_archive_sha256']={str(c):digest(feature_root/f'cutoff_{c}'/'neural_b0.0.npz') for c,_ in plan['folds']}
    plan['author_reference_sha256']=digest(HERE/'STAGE_DYNAMICS_REFERENCE.json')
    plan['dependencies']={k:importlib.metadata.version(k) for k in ['torch','torchdiffeq','geomloss','numpy','scipy','scikit-learn']}
    if args.resume:
        existing=json.loads((out/'plan.json').read_text())
        for key in ['source_sha256','prepared_report_sha256','author_reference_sha256','dependencies','configs','growth_proxy_sha256','feature_archive_sha256']:
            if existing[key]!=plan[key]:raise ValueError('Resume plan mismatch: '+key)
        plan=existing
    else:
        out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2))
    events=out/'events.jsonl'
    if not args.resume:
        for f in plan['source_sha256']:(out/f).write_bytes((HERE/f).read_bytes())
    append_event(events,'neural_plan_resumed' if args.resume else 'matched_horizon_plan_frozen',sha256=digest(out/'plan.json'))
    x=np.load(data/'expression.npy',mmap_mode='r');metadata=pd.read_csv(data/'selected_metadata.csv');stages=metadata.numeric_stage.to_numpy(float)
    symbols=pd.read_csv(data/'genes.csv').symbol.fillna('').tolist()
    panel=(root/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
    counts=Counter(symbols);lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
    official=np.array([i for i,s in enumerate(panel) if s in lookup]);atlas=np.array([lookup[panel[i]] for i in official])
    def values(rows):
        result=np.zeros((len(rows),len(panel)),dtype=np.float32)
        result[:,official]=np.asarray(x[np.ix_(rows,atlas)])
        return result
    core,manifest=load_core();report={'plan':plan,'folds':[],'official_score':None,'local_gate_passed':False,
        'scorer_manifest_sha256':digest(HERE/'private/scorer_source/manifest.json')}
    if args.resume and (out/'report.partial.json').exists():report=json.loads((out/'report.partial.json').read_text());report.pop('in_progress_fold',None)
    for cutoff,target in plan['folds']:
        if any(f['cutoff']==cutoff for f in report['folds']):continue
        folder=out/f'cutoff_{cutoff}';folder.mkdir(exist_ok=args.resume)
        donor_rows=np.sort(np.random.default_rng(plan['seed']).choice(np.flatnonzero(stages==cutoff),1500,replace=False))
        donors=values(donor_rows);np.save(folder/'donor_rows.npy',donor_rows)
        old_model=feature_root/f'cutoff_{cutoff}'
        if digest(old_model/'neural_b0.0.npz')!=plan['feature_archive_sha256'][str(cutoff)]:raise ValueError('Feature archive changed')
        saved=np.load(old_model/'neural_b0.0.npz');features=saved['features']
        proxy_path=proxy_root/f'scores_{cutoff}.npz'
        if digest(proxy_path)!=plan['growth_proxy_sha256'][str(cutoff)]:raise ValueError('Growth inputs changed')
        proxy=np.load(proxy_path)
        programs={}
        for mode,reg in [(m,r) for m in ['neutral','growth1'] for r in [1.,10.]]:
            model_name=mode+'_r'+str(reg)
            flow=StageTransportFlow(x,stages,cutoff,panel,symbols,features,proxy['rows'],proxy['net'],mode,time_ridge=reg)
            flow.save(folder/(model_name+'_flow.npz'))
            program=FullConditionalPositiveForecast(x,stages,cutoff,donors,panel,symbols,flow.net,flow.center,flow.scale,features)
            program.audit.update(method='Stage-conditioned historical transport flow with full conditional positive-abundance head',latent_flow=flow.audit,
                scope='Past-only time-conditioned affine regression and full conditional positive ridge head; no VAE or CNF reproduction. Future time features extrapolate beyond support; no calibrated biological growth claim.')
            np.savez_compressed(folder/(model_name+'_heads.npz'),detection=program.detection,positive_coef=program.positive_coef,positive_center=program.positive_center,positive_mean=program.positive_mean,pmean=program.pmean,zcenter=program.zcenter,support=program.support)
            programs[(mode,reg)]=program
            append_event(events,'transport_and_hurdle_heads_fit_completed',cutoff=cutoff,audit=program.audit)
        oldfolder=old/f'cutoff_{cutoff}'
        np.testing.assert_array_equal(donor_rows,np.load(oldfolder/'donor_rows.npy'))
        oldgeneration=json.loads((oldfolder/'generation.json').read_text())
        generation={}
        for name in plan['configs']:
            if name.startswith('stage_'):
                _,mode,reg,strength=name.split('_')
                pred,indices,audit=programs[(mode,float(reg[1:]))].predict(target,'abundance',float(strength[1:]))
            else:
                previous=name
                source=oldfolder/(previous+'.npy')
                if digest(source)!=plan['archived_control_predictions_sha256'][f'cutoff_{cutoff}/{previous}.npy']:raise ValueError('Archived prediction changed')
                pred=np.load(source);indices=np.load(oldfolder/(previous+'_indices.npy'))
                audit={**oldgeneration[previous]['audit'],'replayed_from':str(source.relative_to(HERE))}
            np.save(folder/(name+'.npy'),pred);np.save(folder/(name+'_indices.npy'),indices)
            generation[name]={'audit':audit,'prediction_sha256':digest(folder/(name+'.npy'))}
            append_event(events,'matched_horizon_forecast_generated',cutoff=cutoff,candidate=name)
            del pred
        (folder/'generation.json').write_text(json.dumps(generation,indent=2))
        append_event(events,'all_fold_predictions_frozen_before_target_read',cutoff=cutoff,target=target)
        target_rows=np.sort(np.random.default_rng(plan['seed']).choice(np.flatnonzero(stages==target),2000,replace=False))
        future=values(target_rows);order=np.random.default_rng(plan['seed']).permutation(2000)
        evaluator=Panel(core,future[order[:1000]],donors,plan['seed'])
        floor=evaluator.metrics(donors);ceiling=evaluator.metrics(future[order[1000:]])
        oldreport=json.loads((old/'report.json').read_text())
        oldfold=next(f for f in oldreport['folds'] if f['cutoff']==cutoff)
        if floor!=oldfold['floor'] or ceiling!=oldfold['ceiling']:raise ValueError('Calibration panels differ')
        results=[]
        for name in plan['configs']:
            pred=np.load(folder/(name+'.npy'),mmap_mode='r');raw=evaluator.metrics(pred)
            result={'candidate':name,'raw_metrics':raw,**evaluator.aggregate(raw,floor,ceiling)}
            results.append(result);append_event(events,'matched_horizon_candidate_scored',cutoff=cutoff,
                candidate=name,score=result['local_score'],valid=result['calibration_valid'],raw_metrics=result['raw_metrics'],skills=result['skills'])
            (out/'report.partial.json').write_text(json.dumps({**report,'in_progress_fold':{'cutoff':cutoff,'target':target,'floor':floor,'ceiling':ceiling,'results':results,'generation':generation}},indent=2))
            del pred
        report['folds'].append({'cutoff':cutoff,'target':target,'floor':floor,'ceiling':ceiling,
            'results':results,'generation':generation})
        (out/'report.partial.json').write_text(json.dumps(report,indent=2))
        del programs,program,flow,evaluator,future,donors
    report['status']='completed';report['decision']='Retain complete results; diagnostic evidence only, no future export or submission.'
    (out/'report.json').write_text(json.dumps(report,indent=2))
    append_event(events,'matched_horizon_audit_completed',evaluations=26)


if __name__=='__main__':
    try:
        torch.set_num_threads(2)
        with threadpool_limits(limits=2):main()
    except Exception as exc:
        failure=HERE/'private/stage_transport_horizon_01/failure.json'
        if failure.parent.exists():
            failure.write_text(json.dumps({'status':'execution_failed','error_type':type(exc).__name__,'error':str(exc),'resume_command':'outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/stage_transport_horizon.py --resume'},indent=2))
        raise
