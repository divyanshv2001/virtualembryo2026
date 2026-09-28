"""Cell-level NMF consensus usage dynamics on two one-day folds."""
import json
from collections import Counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from detection_transfer import DetectionTransfer
from cell_program_trend import CellProgramTrend
from train_extended_atlas import HERE
from offline_backtest import load_core, Panel
from run_t1 import digest
from iterate import now, append_event


def main():
    root=HERE.parents[1];data=HERE/'private/associated_prepared_01'
    out=HERE/'private/cell_program_horizon_pilot_01'
    if out.exists():raise ValueError('Preserve previous run; no overwrite')
    prepared=json.loads((data/'report.json').read_text())
    for f,k in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(data/f)!=prepared[k]:raise ValueError('Prepared data changed')
    plan={'created_before_training_utc':now(),'folds':[[8.,9.],[8.25,9.25]],
        'configs':['copy','unit16']+[f'cell_r{r}_d{d}' for r in [8,16] for d in [0.,.5]],
        'donor_count':1500,'truth_count':1000,'ceiling_count':1000,'seed':20260928,
        'prepared_report_sha256':digest(data/'report.json'),
        'source_sha256':{f:digest(HERE/f) for f in ['cell_program_horizon_pilot.py','detection_transfer.py',
            'challenge_transfer.py','robust_population.py','constrained_forecast.py','transfer_genes.py','offline_backtest.py','annotation_trend.py','cell_program_trend.py']},
        'fit':'Same associated cohort and unit16, identity alignment, expression factor cap1.25, covariance limit.4. Each fold fits only rows <=cutoff.',
        'ablation':'Annotations define observed cell types. Latest three past snapshots, minimum20 cells per type per stage, minimum10 positive cells for positive-mean slope. Stable sign-consistent slope and n/(n+64) shrinkage. Fixed population. Cell-level NMF ranks8/16, three fixed replicas, max200 iterations/tol.001, 3000 past fit cells,384 variance-selected features. Scaled normalized abundance, median Kmeans consensus without outlier filtering. Fit fixed-component usages for all past cells, normalize usages, ridge1 full-panel log-abundance decoder. Stable within-type usage slopes. Forecast with detection0 or.5. Factor cap1.25, probability cap.02, covariance guard.4.',
        'scope':'One-day source-cohort development, not independent challenge tissue replication. Atlas donor/target missing or ambiguous official genes are zero placeholders, not observed zeros. This experiment also uses observed annotation strings as learner grouping features. Labels may reflect joint atlas processing, so this is retrospective annotation-conditioned development, not a clean independently annotated holdout. Later-stage label values never enter fitting or prediction.',
        'gate':'Diagnostic batch only; cannot pass >72 final promotion or certify hidden E10.5. Keep failed official model as local control.',
        'literature':'https://pmc.ncbi.nlm.nih.gov/articles/PMC6639075/; preprocessing, consensus methods and simulation benchmark read. Adaptation, not cNMF reproduction: normalized abundance,384 genes,3 replicas, no outlier filter, added decoder and dynamics.',
        'submissions_allowed':0,'jev_requests_allowed':0}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2));events=out/'events.jsonl'
    for f in plan['source_sha256']:(out/f).write_bytes((HERE/f).read_bytes())
    append_event(events,'matched_horizon_plan_frozen',sha256=digest(out/'plan.json'))
    x=np.load(data/'expression.npy',mmap_mode='r');metadata=pd.read_csv(data/'selected_metadata.csv');stages=metadata.numeric_stage.to_numpy(float);source_labels=metadata.celltype_extended_atlas.astype(str).to_numpy()
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
    for cutoff,target in plan['folds']:
        folder=out/f'cutoff_{cutoff}';folder.mkdir()
        donor_rows=np.sort(np.random.default_rng(plan['seed']).choice(np.flatnonzero(stages==cutoff),1500,replace=False))
        donors=values(donor_rows);np.save(folder/'donor_rows.npy',donor_rows)
        model=DetectionTransfer(x,stages,cutoff,donors,panel,symbols,states=16,alignment='identity',
            feature_scaling='unit',covariance_limit=.4,expression_factor_cap=1.25)
        model.save(folder/'model.npz')
        programs={r:CellProgramTrend(x,stages,source_labels,cutoff,donors,source_labels[donor_rows],panel,symbols,model.official_features,rank=r) for r in [8,16]}
        for r,program in programs.items():program.save(folder/f'cell_program_r{r}.npz')
        generation={}
        for name in plan['configs']:
            if name=='copy':pred=donors.copy();indices=np.arange(len(donors));audit={'method':'persistence'}
            elif name=='unit16':pred,indices,audit=model.predict_detection(target,.5,1.,.02)
            else:
                rank,det=name.split('_')[1:]
                pred,indices,audit=programs[int(rank[1:])].predict_cell_program(target,float(det[1:]))
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
        results=[]
        for name in plan['configs']:
            pred=np.load(folder/(name+'.npy'),mmap_mode='r');raw=evaluator.metrics(pred)
            result={'candidate':name,'raw_metrics':raw,**evaluator.aggregate(raw,floor,ceiling)}
            results.append(result);append_event(events,'matched_horizon_candidate_scored',cutoff=cutoff,
                candidate=name,score=result['local_score'],valid=result['calibration_valid'])
            del pred
        report['folds'].append({'cutoff':cutoff,'target':target,'floor':floor,'ceiling':ceiling,
            'results':results,'generation':generation})
        (out/'report.partial.json').write_text(json.dumps(report,indent=2))
        del model,programs,evaluator,future,donors
    report['status']='completed';report['decision']='Retain complete results; diagnostic evidence only, no future export or submission.'
    (out/'report.json').write_text(json.dumps(report,indent=2))
    append_event(events,'matched_horizon_audit_completed',evaluations=12)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
