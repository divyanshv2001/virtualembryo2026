"""Frozen, past-only DE-rank trend correction on reused E9.5 development panels."""
import json
import importlib.metadata
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event
from hurdle_backtest import read_cells
from offline_backtest import load_core, Panel
from past_gene_rank_correction import fit_consistent_gene_trends, apply_positive_only_correction
from temporary_forecast_cache import TemporaryForecastCache


def main():
    root=HERE.parents[1];out=HERE/'private/de_rank_trend_challenge_01'
    if out.exists():raise ValueError('Preserve previous trial')
    source=HERE/'private/associated_prepared_01'
    archive=HERE/'private/cnf_feature_challenge_01'
    anchor_path=root/'data/E8.5_RNA.h5ad';target_path=root/'data/E9.5_RNA.h5ad'
    panel_path=root/'outputs/t1_run/T1__val.genes.txt';panel=panel_path.read_text().splitlines()
    baseline_path=archive/'unit16.npy'
    seeds=[20260928,20260929,20260930]
    candidates=['copy','unit16']+[f'trend_k{k}_s{s}' for k in [64,256] for s in [.5,1.]]
    code=['de_rank_trend_challenge.py','past_gene_rank_correction.py','temporary_forecast_cache.py',
          'offline_backtest.py','hurdle_backtest.py']
    plan={'created_utc':now(),'source_sha256':{f:digest(HERE/f) for f in code},
          'input_sha256':{str(p.relative_to(root)):digest(p) for p in [anchor_path,target_path,panel_path]},
          'prepared_report_sha256':digest(source/'report.json'),
          'archived_baseline_sha256':digest(baseline_path),
          'archived_report_sha256':digest(archive/'report.json'),
          'dependencies':{k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-learn','anndata']},
          'cutoff':8.5,'target':9.5,'donor_seed':20260928,'seeds':seeds,
          'donor_count':1500,'truth_count':1000,'ceiling_count':1000,'candidates':candidates,
          'hypothesis':'Consistent signed gene trends over permitted atlas E8.0,8.25,8.5 may improve the true changed-gene ranking missed by broad latent forecasts.',
          'fit':'Past-only per-gene log1p pseudobulk trends in three 3,000-cell source stages. Retain same-sign consecutive quarter-day changes, sufficient anchor detection, rank by weaker absolute change, extrapolate at most 0.15 log units for one day. Apply only to positive entries of archived unit16 forecast; no E9.5 fitting.',
          'ablation':'Top 64/256 genes crossed with half/full correction. Matched unit16 and donor-copy controls, three frozen reused development panels; 18 complete scores.',
          'scope':'Unchanged 32,285-gene scorer and original floor/ceiling; source atlas sample and reused E9.5 panels are development evidence only. Top-k is chosen before target read, not optimized from target DE labels.',
          'retention':'D-only short-lived forecast TemporaryFile cache; one matrix loaded for scoring at a time. Preserve small plan, row IDs, selected gene offsets, hashes, metrics and genuine events.',
          'gate':'No official submission. >72 mean/lower-tail 64 Monte Carlo and temporal transfer readiness unchanged.',
          'submissions_allowed':0,'jev_requests_allowed':0}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2))
    events=out/'events.jsonl';append_event(events,'plan_frozen',sha256=digest(out/'plan.json'))
    path=HERE/'LOCAL_OPTIMIZATION_STATE.json';state=json.loads(path.read_text())
    state.setdefault('de_rank_trend_job',{}).update(status='running',run=out.name,evaluations_planned=18,
        evaluations=0,pending_evaluations=18,plan_sha256=digest(out/'plan.json'),
        storage='D-only temporary cache; no retained full forecast matrices')
    state['active_jobs']=[out.name];state['active_run_path']=f'private/{out.name}'
    state['local_process_running']=True
    state['next_experiment']='Past-only DE gene-rank trend trial in progress; inspect report before starting another run.'
    path.write_text(json.dumps(state,indent=2))
    path=HERE/'METRIC_RESEARCH_QUEUE.json';queue=json.loads(path.read_text())
    queue['paths'].append({'id':'de_consistent_past_gene_rank_prior','run':out.name,'status':'running',
        'implementation':['past_gene_rank_correction.py','de_rank_trend_challenge.py'],
        'metric_objective':['de_score','de_direction','mmd_u','variogram'],
        'hypothesis':plan['hypothesis'],'trial_scope':plan['scope'],
        'limitations':'Own bounded trend prior, not DESeq2/ashr reproduction; source-stage associations and reused development do not establish hidden transfer.',
        'research_family_exhausted':False})
    path.write_text(json.dumps(queue,indent=2))
    prepared=json.loads((source/'report.json').read_text())
    for name,key in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(source/name)!=prepared[key]:raise ValueError('Prepared source changed: '+name)
    x=np.load(source/'expression.npy',mmap_mode='r')
    stages=pd.read_csv(source/'selected_metadata.csv').numeric_stage.to_numpy(float)
    symbols=pd.read_csv(source/'genes.csv').symbol.fillna('').tolist()
    donors,donor_rows=read_cells(anchor_path,panel,1500,plan['donor_seed'])
    np.testing.assert_array_equal(donor_rows,np.load(archive/'anchor_rows.npy'))
    np.save(out/'anchor_rows.npy',donor_rows)
    if digest(baseline_path)!=plan['archived_baseline_sha256']:raise ValueError('Archived baseline changed')
    baseline=np.load(baseline_path,mmap_mode='r')
    if baseline.shape!=donors.shape or baseline.dtype!=np.float32:raise ValueError('Unit16 baseline mismatch')
    selections={}
    for k in [64,256]:
        genes,offset,audit=fit_consistent_gene_trends(x,stages,panel,symbols,donors,k)
        np.savez_compressed(out/f'trend_k{k}.npz',genes=genes,offset=offset)
        selections[k]=(genes,offset,audit)
        append_event(events,'past_trend_fitted',top_k=k,selection_sha256=digest(out/f'trend_k{k}.npz'),audit=audit)
    cache=TemporaryForecastCache(HERE/'private/temporary_cache')
    try:
        generation={}
        for name in candidates:
            if name=='copy':
                pred=donors.copy();indices=np.arange(len(donors));audit={'method':'Measured E8.5 persistence'}
            elif name=='unit16':
                pred=np.asarray(baseline);indices=np.load(archive/'unit16_indices.npy')
                audit={'method':'Archived frozen unit16 baseline','archive_sha256':plan['archived_baseline_sha256']}
            else:
                k=int(name.split('_')[1][1:]);strength=float(name.split('_')[2][1:])
                genes,offset,trend_audit=selections[k]
                pred=apply_positive_only_correction(baseline,genes,offset,strength)
                indices=np.load(archive/'unit16_indices.npy')
                audit={'method':'Consistent past gene-rank prior','strength':strength,**trend_audit}
            np.save(out/(name+'_indices.npy'),indices)
            generation[name]={'prediction_sha256':cache.put(name,pred),'audit':audit,
                              'retention':'temporary_D_cache_until_scored'}
            append_event(events,'forecast_frozen',candidate=name,prediction_sha256=generation[name]['prediction_sha256'])
            del pred
        (out/'generation.json').write_text(json.dumps(generation,indent=2))
        append_event(events,'all_forecasts_frozen_before_target_read')
        core,_=load_core();old=json.loads((archive/'report.json').read_text())
        report={'plan':plan,'generation':generation,'panels':[],
                'scorer_manifest_sha256':digest(HERE/'private/scorer_source/manifest.json'),
                'official_score':None,'local_gate_passed':False}
        for seed in seeds:
            future,rows=read_cells(target_path,panel,2000,seed)
            np.save(out/f'target_rows_{seed}.npy',rows)
            order=np.random.default_rng(seed).permutation(len(future))
            evaluator=Panel(core,future[order[:1000]],donors,seed)
            floor=evaluator.metrics(donors);ceiling=evaluator.metrics(future[order[1000:]])
            prior=next(p for p in old['panels'] if p['seed']==seed)
            if floor!=prior['floor'] or ceiling!=prior['ceiling']:raise ValueError('Calibration changed')
            results=[]
            for name in candidates:
                with cache.read(name,consume=False) as pred:raw=evaluator.metrics(pred)
                result={'candidate':name,'prediction_sha256':generation[name]['prediction_sha256'],
                        'raw_metrics':raw,**evaluator.aggregate(raw,floor,ceiling)}
                results.append(result);append_event(events,'candidate_scored',seed=seed,**result)
            report['panels'].append({'seed':seed,'cutoff':8.5,'target':9.5,
                                     'floor':floor,'ceiling':ceiling,'results':results})
            (out/'report.partial.json').write_text(json.dumps(report,indent=2))
        report['status']='completed';(out/'report.json').write_text(json.dumps(report,indent=2))
        summaries=[]
        for name in candidates:
            rows=[next(r for r in p['results'] if r['candidate']==name) for p in report['panels']]
            valid=all(r['calibration_valid'] for r in rows)
            summaries.append({'candidate':name,'scores':[r['local_score'] for r in rows],
                'mean_score':float(np.mean([r['local_score'] for r in rows])) if valid else None,
                'raw_metrics':[r['raw_metrics'] for r in rows],
                'skills':[r['skills'] for r in rows],'all_calibrations_valid':valid})
        public={'updated_utc':now(),'status':'completed_not_promoted','summaries':summaries,
            'panels':report['panels'],'plan_sha256':digest(out/'plan.json'),
            'report_sha256':digest(out/'report.json'),'scope':plan['scope'],
            'local_72_gate_passed':False,'official_score':None,'submissions_used':0}
        (HERE/'DE_RANK_TREND_RESULTS.json').write_text(json.dumps(public,indent=2))
        state=json.loads((HERE/'LOCAL_OPTIMIZATION_STATE.json').read_text())
        state['de_rank_trend_job'].update(status='completed_not_promoted',evaluations=18,
            pending_evaluations=0,report_sha256=public['report_sha256'],results_report='DE_RANK_TREND_RESULTS.json')
        state['active_jobs']=[];state['local_process_running']=False
        (HERE/'LOCAL_OPTIMIZATION_STATE.json').write_text(json.dumps(state,indent=2))
        queue=json.loads((HERE/'METRIC_RESEARCH_QUEUE.json').read_text())
        for entry in queue['paths']:
            if entry.get('run')==out.name:entry.update(status='implemented_evaluated_not_promoted',results_report='DE_RANK_TREND_RESULTS.json')
        (HERE/'METRIC_RESEARCH_QUEUE.json').write_text(json.dumps(queue,indent=2))
        from index_scores import main as index_scores
        index_scores()
        from update_compact_checkpoint import refresh
        refresh('Completed 18 full-panel past gene-rank trend scores with matched controls; inspect raw DE/MMD/variogram and paired outcomes before any promotion.')
        append_event(events,'trial_completed',evaluations=18)
    finally:cache.close()


if __name__=='__main__':
    try:
        with threadpool_limits(limits=2):main()
    except Exception as exc:
        folder=HERE/'private/de_rank_trend_challenge_01'
        if folder.exists():
            (folder/'failure.json').write_text(json.dumps({'status':'execution_failed',
                'error_type':type(exc).__name__,'error':str(exc)},indent=2))
            append_event(folder/'events.jsonl','execution_failed',error_type=type(exc).__name__,error=str(exc))
        raise
