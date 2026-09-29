"""Past-only kernel population drift on fixed flow forecasts; development only."""
import json
from collections import Counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now,append_event
from hurdle_backtest import read_cells
from kernel_population import KernelPopulation
from offline_backtest import load_core,Panel


def main():
    root=HERE.parents[1];out=HERE/'private/cnf_kernel_population_challenge_01'
    if out.exists():raise ValueError('Preserve existing run; no silent overwrite')
    source=HERE/'private/cnf_anchor_slope_challenge_01';feature=HERE/'private/cnf_feature_challenge_01'
    data=HERE/'private/associated_prepared_01';prepared=json.loads((data/'report.json').read_text())
    for f,k in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(data/f)!=prepared[k]:raise ValueError('Prepared data changed')
    configs=[(base,strength,penalty) for base in ['copy','anchorslope_both_0.5'] for strength in [.5,1.] for penalty in [.01,.1]]
    def label(c):return f'kernel_{c[0]}_s{c[1]}_p{c[2]}'
    controls=['copy','unit16','anchorslope_both_0.5']
    panel_path=root/'outputs/t1_run/T1__val.genes.txt';panel=panel_path.read_text().splitlines()
    anchor=root/'data/E8.5_RNA.h5ad';target=root/'data/E9.5_RNA.h5ad'
    code=['cnf_kernel_population_challenge.py','kernel_population.py','constrained_forecast.py','robust_population.py','offline_backtest.py','hurdle_backtest.py']
    plan={'created_before_fitting_utc':now(),'configs':controls+[label(c) for c in configs],'cutoff':8.5,'target':9.5,'evaluation_seeds':[20260928,20260929,20260930],'donor_seed':20260928,'donor_count':1500,'truth_count':1000,'ceiling_count':1000,
        'source_sha256':{f:digest(HERE/f) for f in code},'prepared_report_sha256':digest(data/'report.json'),'encoder_sha256':digest(feature/'encoder4096.npz'),'input_sha256':{p.name:digest(p) for p in [anchor,target,panel_path]},
        'archived_predictions_sha256':{n:digest(source/(n+'.npy')) for n in controls},'archived_report_sha256':digest(source/'report.json'),
        'fit':'Reuse4096gene8D whitened past CNF encoder. Population projector clips standardized inputs to10 as existing kernel method; disclosed difference from flow encoder. Forecast lastfive source-stage RFF means from<=E8.5. Entropy regularization and capped logits; covariance<=.4 and ESS>=.6N.500RFF dimensions, fixed frequencies/seeds. No target fitting.',
        'hypothesis':'Reweighting whole fixed forecasts may repair population proportions without changing conditional gene shifts. Prior kernel branch failed with unit16 base; this bounded integration tests new fixed slope base versus empirical copy.',
        'scope':'Eight weighting variants plus three archives,33 full32285gene development scores. Clipped projector is a population surrogate, never a scorer replacement. No independent embryos or full atlas training. Captured composition is not measured biological growth.',
        'coverage':'Own RFF/entropy adaptation of previously reviewed kernel matching literature in METRIC_RESEARCH_QUEUE.mmd_kernel_matching; not published KMM reproduction.',
        'gate':'No upload/Jev/agents; original>72 mean/lower-tail64replicate and temporal gates unchanged.'}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2))
    for f in code:(out/f).write_bytes((HERE/f).read_bytes())
    events=out/'events.jsonl';append_event(events,'population_plan_frozen',plan_sha256=digest(out/'plan.json'))
    x=np.load(data/'expression.npy',mmap_mode='r');stages=pd.read_csv(data/'selected_metadata.csv').numeric_stage.to_numpy(float)
    symbols=pd.read_csv(data/'genes.csv').symbol.fillna('').tolist();counts=Counter(symbols);lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
    saved=np.load(feature/'encoder4096.npz');official=saved['features'];atlas=np.array([lookup[panel[i]] for i in official])
    donors,rows=read_cells(anchor,panel,1500,20260928);np.testing.assert_array_equal(rows,np.load(source/'anchor_rows.npy'));np.save(out/'anchor_rows.npy',rows)
    encoder={'cutoff':np.array(8.5),'alignment':np.array('identity'),'official_features':official,'features':atlas,'center':saved['center'],'scale':saved['scale'],'pca_mean':saved['pca_center'],'pca_components':saved['basis'],'trusted':np.ones(len(donors),dtype=bool)}
    # No support filter is invented: population support is bounded by ESS/covariance.
    generation={};previous=json.loads((source/'generation.json').read_text())
    for n in controls:
        if digest(source/(n+'.npy'))!=plan['archived_predictions_sha256'][n]:raise ValueError('Archive changed')
        pred=np.load(source/(n+'.npy'));indices=np.load(source/(n+'_indices.npy'))
        np.save(out/(n+'.npy'),pred);np.save(out/(n+'_indices.npy'),indices)
        generation[n]={'prediction_sha256':digest(out/(n+'.npy')),'audit':previous[n]['audit']}
    for base in ['copy','anchorslope_both_0.5']:
        frozen=np.load(source/(base+'.npy'));indices=np.load(source/(base+'_indices.npy'))
        model=KernelPopulation(x,stages,8.5,encoder,donors,frozen,indices,dimensions=500);model.save(out/(base+'_population.npz'))
        append_event(events,'past_only_kernel_fitted',base=base,model_sha256=digest(out/(base+'_population.npz')))
        for c in [c for c in configs if c[0]==base]:
            n=label(c);pred,origin,audit=model.predict(9.5,c[1],c[2],window=5)
            np.save(out/(n+'.npy'),pred);np.save(out/(n+'_indices.npy'),origin)
            generation[n]={'prediction_sha256':digest(out/(n+'.npy')),'audit':audit}
            append_event(events,'population_forecast_frozen',candidate=n,**generation[n])
    (out/'generation.json').write_text(json.dumps(generation,indent=2));append_event(events,'all_predictions_frozen_before_target_read',candidates=plan['configs'])
    core,_=load_core();report={'plan':plan,'panels':[],'status':'running','official_score':None,'local_gate_passed':False,'scorer_manifest_sha256':digest(HERE/'private/scorer_source/manifest.json')}
    reference=json.loads((source/'report.json').read_text())
    for seed in plan['evaluation_seeds']:
        future,rows=read_cells(target,panel,2000,seed);np.save(out/f'target_rows_{seed}.npy',rows)
        order=np.random.default_rng(seed).permutation(2000);evaluator=Panel(core,future[order[:1000]],donors,seed)
        floor=evaluator.metrics(donors);ceiling=evaluator.metrics(future[order[1000:]])
        old=next(p for p in reference['panels'] if p['seed']==seed)
        if floor!=old['floor'] or ceiling!=old['ceiling']:raise ValueError('Calibration changed')
        results=[]
        for n in plan['configs']:
            if digest(out/(n+'.npy'))!=generation[n]['prediction_sha256']:raise ValueError('Prediction changed')
            raw=evaluator.metrics(np.load(out/(n+'.npy'),mmap_mode='r'));result={'candidate':n,'raw_metrics':raw,**evaluator.aggregate(raw,floor,ceiling)}
            results.append(result);append_event(events,'population_candidate_scored',seed=seed,**result)
            (out/'report.partial.json').write_text(json.dumps({**report,'in_progress_panel':{'seed':seed,'cutoff':8.5,'target':9.5,'floor':floor,'ceiling':ceiling,'results':results,'generation':generation}},indent=2))
        report['panels'].append({'seed':seed,'cutoff':8.5,'target':9.5,'floor':floor,'ceiling':ceiling,'results':results,'generation':generation})
        (out/'report.partial.json').write_text(json.dumps(report,indent=2))
    report['status']='completed';(out/'report.json').write_text(json.dumps(report,indent=2))
    summaries=[]
    for n in plan['configs']:
        results=[next(r for r in p['results'] if r['candidate']==n) for p in report['panels']];valid=all(r['calibration_valid'] for r in results)
        summaries.append({'candidate':n,'scores':[r['local_score'] for r in results],'mean_score':float(np.mean([r['local_score'] for r in results])) if valid else None,'all_calibrations_valid':valid,'raw_metrics':[r['raw_metrics'] for r in results],'skills':[r['skills'] for r in results]})
    public={'status':'completed_not_promoted','updated_utc':now(),'summaries':summaries,'panels':report['panels'],'plan_sha256':digest(out/'plan.json'),'report_sha256':digest(out/'report.json'),'scope':plan['scope'],'local_72_gate_passed':False,'official_score':None,'submissions_used':0}
    (HERE/'CNF_KERNEL_POPULATION_RESULTS.json').write_text(json.dumps(public,indent=2))
    f=HERE/'LOCAL_OPTIMIZATION_STATE.json';s=json.loads(f.read_text());s['cnf_kernel_population_job'].update(status='completed_not_promoted',evaluations=33,pending_evaluations=0,results_report='CNF_KERNEL_POPULATION_RESULTS.json',report_sha256=public['report_sha256']);s['active_jobs']=[];s['local_process_running']=False;f.write_text(json.dumps(s,indent=2))
    from index_scores import main as index_scores
    index_scores();append_event(events,'population_batch_completed',evaluations=33)


if __name__=='__main__':
    try:
        with threadpool_limits(limits=2):main()
    except Exception as exc:
        out=HERE/'private/cnf_kernel_population_challenge_01'
        if out.exists():(out/'failure.json').write_text(json.dumps({'status':'execution_failed','error_type':type(exc).__name__,'error':str(exc)}))
        raise
