"""Past-only marker growth-proxy composition sensitivity, not unbalanced OT."""
import json
import argparse
import importlib.metadata
from collections import Counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from growth_proxy_forecast import GrowthProxyComposition
from train_extended_atlas import HERE
from offline_backtest import load_core, Panel
from run_t1 import digest
from iterate import now, append_event


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    root=HERE.parents[1];data=HERE/'private/associated_prepared_01'
    out=HERE/'private/growth_composition_horizon_01'
    if out.exists() and not args.resume:raise ValueError('Preserve previous run; use --resume')
    prepared=json.loads((data/'report.json').read_text())
    for f,k in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(data/f)!=prepared[k]:raise ValueError('Prepared data changed')
    plan={'created_before_training_utc':now(),'folds':[[8.,9.],[8.25,9.25]],
        'configs':['copy','neutral','unit16','saved384_d0.0']+[f'growth_{k}_p{v}' for k in ['net','proliferation','p53'] for v in [.5,1.]],
        'donor_count':1500,'truth_count':1000,'ceiling_count':1000,'seed':20260928,
        'prepared_report_sha256':digest(data/'report.json'),
        'source_sha256':{f:digest(HERE/f) for f in ['growth_composition_horizon.py','detection_transfer.py',
            'challenge_transfer.py','robust_population.py','constrained_forecast.py','transfer_genes.py','offline_backtest.py','annotation_trend.py','growth_proxy_forecast.py','growth_prior_audit.py']},
        'fit':'Use repaired past-only marker/control scores; one-day growth-proxy weighted resampling of1500 latest-stage donors. Guard features are previously past-selected384 neural features. No new expression or transport dynamics are fitted.',
        'ablation':'Neutral growth versus proliferation-minus-P53, proliferation-only and negative-P53 relative-growth scores. Powers.5/1, latest-donor median center/std>=.1, one-day exponent, relative-weight cap.5-2, common systematic-resampling seed20260928. Covariance guard.4 and recorded projection backoffs. Every output cell copies a complete donor row; library mass and protected values travel with that row.',
        'scope':'Preliminary growth-proxy composition sensitivity, not Waddington-OT/moscot reproduction or unbalanced transport. Mouse P53 list is an uncertain death proxy. Rates are not embryonic birth/death measurements. Equal prepared stage counts are sampling quotas. Marker fitting uses only stages<=cutoff, no annotations. Missing official genes use zero placeholders; two exposed one-day source-cohort development folds cannot certify hidden challenge or independent embryo generalization.',
        'gate':'Diagnostic batch only; cannot pass >72 final promotion or certify hidden E10.5. Keep failed official model as local control.',
        'literature':'https://broadinstitute.github.io/wot/tutorial/ and https://moscot.readthedocs.io/en/stable/notebooks/examples/problems/TemporalProblem/800_score_genes_for_marginals.html; temporal-coupling, prior-marker and growth-sensitivity sections reviewed. Mouse lists and source provenance pinned in GROWTH_MARKER_REFERENCE.json. This preliminary test does not implement author transport solvers.',
        'submissions_allowed':0,'jev_requests_allowed':0}
    old=HERE/'private/neural_ode_horizon_01'
    proxy=HERE/'private/growth_prior_audit_repair_01'
    plan['archived_control_report_sha256']=digest(old/'report.json')
    plan['archived_control_predictions_sha256']={f'cutoff_{c}/{n}.npy':digest(old/f'cutoff_{c}'/(n+'.npy')) for c,_ in plan['folds'] for n in ['copy','unit16','saved384_d0.0']}
    plan['frozen_feature_models_sha256']={str(c):digest(old/f'cutoff_{c}'/'neural_b0.0.npz') for c,_ in plan['folds']}
    plan['growth_score_sha256']={str(c):digest(proxy/f'scores_{c}.npz') for c,_ in plan['folds']}
    plan['growth_audit_report_sha256']=digest(proxy/'report.json')
    plan['marker_reference_sha256']=digest(HERE/'GROWTH_MARKER_REFERENCE.json')
    plan['dependencies']={k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-learn']}
    if args.resume:
        existing=json.loads((out/'plan.json').read_text())
        for key in ['source_sha256','prepared_report_sha256','marker_reference_sha256','dependencies','configs','frozen_feature_models_sha256','growth_score_sha256','growth_audit_report_sha256']:
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
        old_model=old/f'cutoff_{cutoff}'/'neural_b0.0.npz'
        if digest(old_model)!=plan['frozen_feature_models_sha256'][str(cutoff)]:raise ValueError('Frozen feature model changed')
        with np.load(old_model) as saved:features=saved['features']
        source=proxy/f'scores_{cutoff}.npz'
        if digest(source)!=plan['growth_score_sha256'][str(cutoff)]:raise ValueError('Past growth scores changed')
        with np.load(source) as saved:
            positions=np.searchsorted(saved['rows'],donor_rows)
            np.testing.assert_array_equal(saved['rows'][positions],donor_rows)
            program=GrowthProxyComposition(donors,saved['proliferation'][positions],saved['p53_proxy'][positions],features,cutoff)
        append_event(events,'growth_proxy_inputs_verified',cutoff=cutoff,max_score_fit_stage=cutoff)
        oldfolder=old/f'cutoff_{cutoff}'
        np.testing.assert_array_equal(donor_rows,np.load(oldfolder/'donor_rows.npy'))
        oldgeneration=json.loads((oldfolder/'generation.json').read_text())
        generation={}
        for name in plan['configs']:
            if name=='neutral':
                pred,indices,audit=program.predict(target,'net',0.)
                np.testing.assert_array_equal(pred,donors)
            elif name.startswith('growth_'):
                _,kind,power=name.split('_')
                pred,indices,audit=program.predict(target,kind,float(power[1:]))
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
        del program,evaluator,future,donors
    report['status']='completed';report['decision']='Retain complete results; diagnostic evidence only, no future export or submission.'
    (out/'report.json').write_text(json.dumps(report,indent=2))
    append_event(events,'matched_horizon_audit_completed',evaluations=20)


if __name__=='__main__':
    try:
        with threadpool_limits(limits=2):main()
    except Exception as exc:
        failure=HERE/'private/growth_composition_horizon_01/failure.json'
        if failure.parent.exists():
            failure.write_text(json.dumps({'status':'execution_failed','error_type':type(exc).__name__,'error':str(exc),'resume_command':'outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/growth_composition_horizon.py --resume'},indent=2))
        raise
