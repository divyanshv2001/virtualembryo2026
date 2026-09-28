"""Declared earlier atlas rolling forecasts; quarter-day evidence, not one-day certification."""
import json
from collections import Counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from detection_transfer import DetectionTransfer
from nonlinear_transport import NonlinearTransport
from train_extended_atlas import HERE
from offline_backtest import load_core,Panel
from run_t1 import digest
from iterate import now,append_event


def main():
    root=HERE.parents[1]; data=HERE/'private/associated_prepared_01';out=HERE/'private/nonlinear_temporal_01'
    if out.exists(): raise ValueError('Preserve existing run; this pilot has no replay mode')
    prepared=json.loads((data/'report.json').read_text())
    for name,key in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(data/name)!=prepared[key]: raise ValueError('Changed prepared input')
    configs=[('copy',0.,1.25),('incumbent',0.,1.25),('mlp',.25,1.25),('mlp',.5,1.25),('mlp',1.,1.25),('ridge',.5,1.25)]
    plan={'created_before_training_utc':now(),'folds':[[8.,8.25],[8.25,8.5]],'configs':configs,
        'prepared_report_sha256':digest(data/'report.json'),'source_sha256':{n:digest(HERE/n) for n in ['nonlinear_temporal.py','nonlinear_transport.py','detection_transfer.py','challenge_transfer.py','robust_population.py','constrained_forecast.py','transfer_genes.py']},
        'scope':'Each learner sees only rows <= its cutoff. Three or four historical stages; target is quarter-day later. This does not establish one-day generalization or same-configuration final promotion.',
        'full_panel':32285,'submissions_allowed':0,'jev_requests_allowed':0}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2));events=out/'events.jsonl'
    append_event(events,'temporal_plan_frozen',sha256=digest(out/'plan.json'))
    x=np.load(data/'expression.npy',mmap_mode='r');stages=pd.read_csv(data/'selected_metadata.csv').numeric_stage.to_numpy(float)
    symbols=pd.read_csv(data/'genes.csv').symbol.fillna('').tolist();panel=(root/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
    counts=Counter(symbols);lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
    official=np.array([i for i,s in enumerate(panel) if s in lookup]);atlas=np.array([lookup[panel[i]] for i in official])
    def values(rows):
        result=np.zeros((len(rows),len(panel)),dtype=np.float32)
        result[:,official]=np.asarray(x[np.ix_(rows,atlas)])
        return result
    report={'plan':plan,'folds':[],'official_72_verified':False,'same_horizon_temporal_gate_passed':False}
    core,_=load_core()
    for cutoff,target in plan['folds']:
        folder=out/f'cutoff_{cutoff}';folder.mkdir();rng=np.random.default_rng(20260928)
        donor_rows=np.sort(rng.choice(np.flatnonzero(stages==cutoff),1500,replace=False));donors=values(donor_rows)
        base_model=DetectionTransfer(x,stages,cutoff,donors,panel,symbols,states=16,alignment='identity',feature_scaling='unit',covariance_limit=.4)
        base,indices,base_audit=base_model.predict_detection(target,.5,1.,.02)
        base_model.save(folder/'encoder.npz')
        with np.load(folder/'encoder.npz') as f:encoder={k:f[k] for k in f.files}
        model=NonlinearTransport(x,stages,cutoff,encoder,donors,base,indices,panel,symbols);model.save(folder/'flow.npz')
        predictions={};audits={}
        for method,strength,cap in configs:
            name=f'{method}_{strength}'
            if method=='copy':prediction=donors.copy();audit={'method':'persistence'}
            elif method=='incumbent':prediction=base.copy();audit=base_audit
            else:prediction,_,audit=model.predict(target,strength,method,cap)
            predictions[name]=prediction;np.save(folder/(name+'.npy'),prediction);audits[name]={**audit,'prediction_sha256':digest(folder/(name+'.npy'))}
        (folder/'generation.json').write_text(json.dumps(audits,indent=2))
        append_event(events,'all_fold_forecasts_frozen_before_target_read',cutoff=cutoff)
        target_rows=np.sort(rng.choice(np.flatnonzero(stages==target),2000,replace=False));future=values(target_rows);order=rng.permutation(2000)
        evaluator=Panel(core,future[order[:1000]],donors,20260928)
        floor=evaluator.metrics(donors);ceiling=evaluator.metrics(future[order[1000:]])
        results=[]
        for name,prediction in predictions.items():
            raw=evaluator.metrics(prediction);result={'candidate':name,'raw_metrics':raw,**evaluator.aggregate(raw,floor,ceiling)}
            results.append(result);append_event(events,'temporal_candidate_scored',cutoff=cutoff,candidate=name,score=result['local_score'],valid=result['calibration_valid'])
        report['folds'].append({'cutoff':cutoff,'target':target,'floor':floor,'ceiling':ceiling,'results':results,'audits':audits})
        (out/'report.partial.json').write_text(json.dumps(report,indent=2))
        del model,base_model,predictions,evaluator,future
    report['status']='completed';report['decision']='Quarter-day mechanistic pilot only. Require subsequent one-day development and explicit same-configuration temporal evidence; no export.'
    (out/'report.json').write_text(json.dumps(report,indent=2))
    append_event(events,'temporal_pilot_completed')

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
