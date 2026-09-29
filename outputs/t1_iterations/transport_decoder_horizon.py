"""Past-only detection and positive-abundance heads on frozen neural dynamics."""
import json
import argparse
import importlib.metadata
from collections import Counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import torch
from transport_latent_flow import AffineTransportNet
from neural_hurdle_forecast import NeuralHurdleForecast
from train_extended_atlas import HERE
from offline_backtest import load_core, Panel
from run_t1 import digest
from iterate import now, append_event


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    root=HERE.parents[1];data=HERE/'private/associated_prepared_01'
    out=HERE/'private/transport_decoder_horizon_01'
    if out.exists() and not args.resume:raise ValueError('Preserve previous run; use --resume')
    prepared=json.loads((data/'report.json').read_text())
    for f,k in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(data/f)!=prepared[k]:raise ValueError('Prepared data changed')
    plan={'created_before_training_utc':now(),'folds':[[8.,9.],[8.25,9.25]],
        'configs':['copy','unit16','saved384_d0.0']+[f'transport_{m}_s0.5' for m in ['balanced','neutral','growth1']]+[f'decoder_{m}_{head}' for m in ['balanced','neutral','growth1'] for head in ['abundance','detection','systematic']],
        'donor_count':1500,'truth_count':1000,'ceiling_count':1000,'seed':20260928,
        'prepared_report_sha256':digest(data/'report.json'),
        'source_sha256':{f:digest(HERE/f) for f in ['transport_decoder_horizon.py','detection_transfer.py',
            'challenge_transfer.py','robust_population.py','constrained_forecast.py','transfer_genes.py','offline_backtest.py','annotation_trend.py','neural_ode_forecast.py','neural_hurdle_forecast.py','transport_latent_flow.py']},
        'fit':'Replay frozen historical transport flows; recompute all-past hurdle heads and verify bit-identical to their archived arrays. No new transport fit or future checkpoint selection.',
        'ablation':'At fixed forecast strength.5, decompose balanced, neutral-unbalanced and growth1 flow forecasts into abundance-only, detection-only and systematic joint switches. Compare archived independent joint forecasts. All projection guards and panels unchanged.',
        'scope':'Design informed by prior exposed development folds; not fresh blind validation. Frozen PCA/coupling subset budgets remain. Missing official genes are zero placeholders. Independent joint control versus each decoder mechanism isolates decoder tradeoffs; broader transport family remains open.',
        'gate':'No official submission. Unchanged >72 Monte Carlo and temporal validation gate remains required.',
        'literature':'Own mechanistic decoder ablation on the previously documented POT/moscot adaptation; no additional published-method replication claim.',
        'submissions_allowed':0,'jev_requests_allowed':0}
    old=HERE/'private/transport_hurdle_horizon_01'
    plan['archived_control_report_sha256']=digest(old/'report.json')
    controls=[n for n in plan['configs'] if not n.startswith('decoder_')]
    plan['archived_control_predictions_sha256']={f'cutoff_{c}/{n}.npy':digest(old/f'cutoff_{c}'/(n+'.npy')) for c,_ in plan['folds'] for n in controls}
    plan['frozen_flow_head_sha256']={f'cutoff_{c}/{m}_{kind}.npz':digest(old/f'cutoff_{c}'/(m+'_'+kind+'.npz')) for c,_ in plan['folds'] for m in ['balanced','neutral','growth1'] for kind in ['flow','heads']}
    feature_root=HERE/'private/neural_ode_horizon_01'
    plan['feature_archive_sha256']={str(c):digest(feature_root/f'cutoff_{c}'/'neural_b0.0.npz') for c,_ in plan['folds']}
    plan['author_reference_sha256']=digest(HERE/'GROWTH_MARKER_REFERENCE.json')
    plan['dependencies']={k:importlib.metadata.version(k) for k in ['torch','torchdiffeq','geomloss','numpy','scipy','scikit-learn']}
    if args.resume:
        existing=json.loads((out/'plan.json').read_text())
        for key in ['source_sha256','prepared_report_sha256','author_reference_sha256','dependencies','configs','frozen_flow_head_sha256','feature_archive_sha256','archived_control_predictions_sha256']:
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
        feature_path=feature_root/f'cutoff_{cutoff}'/'neural_b0.0.npz'
        if digest(feature_path)!=plan['feature_archive_sha256'][str(cutoff)]:raise ValueError('Feature archive changed')
        features=np.load(feature_path)['features'];old_model=old/f'cutoff_{cutoff}'
        original_generation=json.loads((old_model/'generation.json').read_text())
        programs={}
        for mode in ['balanced','neutral','growth1']:
            for kind in ['flow','heads']:
                path=old_model/(mode+'_'+kind+'.npz')
                if digest(path)!=plan['frozen_flow_head_sha256'][f'cutoff_{cutoff}/{mode}_{kind}.npz']:raise ValueError('Frozen transport inputs changed')
            flow=np.load(old_model/(mode+'_flow.npz'))
            net=AffineTransportNet(flow['basis'],flow['pca_center'],flow['coefficient'])
            program=NeuralHurdleForecast(x,stages,cutoff,donors,panel,symbols,net,flow['center'],flow['scale'],features)
            with np.load(old_model/(mode+'_heads.npz')) as heads:
                for head in ['detection','positive_coef','positive_center','positive_mean','pmean','zcenter','support']:
                    np.testing.assert_array_equal(getattr(program,head),heads[head])
            prior=original_generation[f'transport_{mode}_s0.5']['audit']
            program.audit.update(method='Frozen PCA transport flow with verified hurdle heads',latent_flow=prior['latent_flow'],
                scope='Target-informed development ablation, not blind validation. Recomputed heads exactly match archive; no VAE dynamics or new transport fitting.')
            programs[mode]=program
            append_event(events,'frozen_transport_heads_verified',cutoff=cutoff,mode=mode,audit=program.audit)
        oldfolder=old/f'cutoff_{cutoff}'
        np.testing.assert_array_equal(donor_rows,np.load(oldfolder/'donor_rows.npy'))
        oldgeneration=json.loads((oldfolder/'generation.json').read_text())
        generation={}
        for name in plan['configs']:
            if name.startswith('decoder_'):
                _,mode,head=name.split('_')
                pred,indices,audit=programs[mode].predict(target,'joint' if head=='systematic' else head,.5,sampling='systematic' if head=='systematic' else 'independent')
            else:
                source=oldfolder/(name+'.npy')
                if digest(source)!=plan['archived_control_predictions_sha256'][f'cutoff_{cutoff}/{name}.npy']:raise ValueError('Archived prediction changed')
                pred=np.load(source);indices=np.load(oldfolder/(name+'_indices.npy'))
                audit={**oldgeneration[name]['audit'],'replayed_from':str(source.relative_to(HERE))}
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
    append_event(events,'matched_horizon_audit_completed',evaluations=30)


if __name__=='__main__':
    try:
        torch.set_num_threads(2)
        with threadpool_limits(limits=2):main()
    except Exception as exc:
        failure=HERE/'private/transport_decoder_horizon_01/failure.json'
        if failure.parent.exists():
            failure.write_text(json.dumps({'status':'execution_failed','error_type':type(exc).__name__,'error':str(exc),'resume_command':'outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/transport_decoder_horizon.py --resume'},indent=2))
        raise
