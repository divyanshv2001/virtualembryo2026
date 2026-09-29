"""Frozen full-E8.5 anchor calibration trial on reused E9.5 development panels."""
import json
import importlib.metadata
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event
from hurdle_backtest import read_cells
from cnf_manifold_flow import DensityFlowNet
from full_anchor_slope_forecast import FullAnchorSlopeForecast
from temporary_forecast_cache import TemporaryForecastCache
from offline_backtest import load_core, Panel


def main():
    root = HERE.parents[1]
    out = HERE/'private/cnf_full_anchor_challenge_01'
    if out.exists():
        raise ValueError('Preserve frozen previous run')
    anchor_path = root/'data/E8.5_RNA.h5ad'
    target_path = root/'data/E9.5_RNA.h5ad'
    source = HERE/'private/associated_prepared_01'
    old = HERE/'private/cnf_feature_challenge_01'
    control = HERE/'private/cnf_log1p_positive_challenge_01/log1p_joint_a0.5.npy'
    panel_path = root/'outputs/t1_run/T1__val.genes.txt'
    panel = panel_path.read_text().splitlines()
    seeds = [20260928, 20260929, 20260930]
    candidates = ['copy', 'log1p_joint_a0.5', 'full_anchor_p0.25_d0.75', 'full_anchor_p0.5_d0.5']
    source_files = ['cnf_full_anchor_challenge.py','full_anchor_slope_forecast.py','partial_anchor_forecast.py',
                    'log1p_positive_forecast.py','anchor_slope_calibration.py','feature_panel_forecast.py',
                    'ridge_conditional_head.py','temporary_forecast_cache.py','offline_backtest.py']
    plan = {'created_utc':now(), 'source_sha256':{f:digest(HERE/f) for f in source_files},
            'input_sha256':{str(p.relative_to(root)):digest(p) for p in [anchor_path,target_path,panel_path]},
            'archive_sha256':{str(p.relative_to(HERE)):digest(p) for p in [old/'encoder4096.npz',old/'features4096.pt',old/'features.npy',control]},
            'prepared_report_sha256':digest(source/'report.json'),
            'dependencies':{k:importlib.metadata.version(k) for k in ['numpy','scipy','torch','scikit-learn','anndata']},
            'seeds':seeds,'candidates':candidates,'cutoff':8.5,'target':9.5,'donor_seed':20260928,
            'donor_count':1500,'truth_count':1000,'ceiling_count':1000,
            'hypothesis':'All 16,787 permitted E8.5 cells stabilize conditional positive/detection slopes compared with 1,500 evaluation donors.',
            'fit':'Frozen past-only 8D encoder/800-step field/source log1p heads. Fit anchor conditional ridge1 positive and detection slopes on all E8.5 cells; use the same 1,500 donor cells for predictions. No E9.5 fitting.',
            'ablation':'Predeclared positive/detection strengths (.25,.75) and (.5,.5); matched donor-copy and archived 1,500-anchor log1p joint control.',
            'scope':'Three reused E9.5 development panels, unchanged 32,285-gene scorer, original floor/ceiling and seeds. Cross-sectional slope adaptation is not a temporal velocity estimate or independent validation.',
            'retention':'Full forecasts in short-lived D-only TemporaryFile cache. Preserve small reports, hashes, indices and events; one forecast loaded for scoring at a time.',
            'gate':'No official upload or export. Existing >=64-Monte-Carlo mean/lower-tail >72 plus temporal readiness remain required.',
            'submissions_allowed':0,'jev_requests_allowed':0}
    out.mkdir()
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    events = out/'events.jsonl'
    append_event(events,'plan_frozen',sha256=digest(out/'plan.json'))
    path=HERE/'LOCAL_OPTIMIZATION_STATE.json';state=json.loads(path.read_text())
    state.setdefault('cnf_full_anchor_job',{}).update(
        status='running',run=out.name,evaluations_planned=12,evaluations=0,
        pending_evaluations=12,storage='D-only temporary forecast cache; no retained full matrices',
        plan_sha256=digest(out/'plan.json'))
    state['active_jobs']=[out.name];state['active_run_path']=f'private/{out.name}'
    state['local_process_running']=True
    state['next_experiment']='Full E8.5 anchor conditional slope calibration trial in progress; inspect frozen report before starting anything else.'
    path.write_text(json.dumps(state,indent=2))
    path=HERE/'METRIC_RESEARCH_QUEUE.json';queue=json.loads(path.read_text())
    if not any(p.get('run')==out.name for p in queue['paths']):
        queue['paths'].append({'id':'continuous_flow_full_anchor_conditional_calibration',
            'run':out.name,'status':'running','implementation':['full_anchor_slope_forecast.py','cnf_full_anchor_challenge.py'],
            'metric_objective':['de_score','de_direction','mmd_u','variogram'],
            'hypothesis':plan['hypothesis'],'trial_scope':plan['scope'],
            'limitations':'Reused development panels; cross-sectional associations not temporal velocities. Bounded trial does not exhaust family.',
            'research_family_exhausted':False})
        path.write_text(json.dumps(queue,indent=2))
    prepared = json.loads((source/'report.json').read_text())
    for name,key in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(source/name) != prepared[key]: raise ValueError('Prepared input changed: '+name)
    x = np.load(source/'expression.npy',mmap_mode='r')
    stages = pd.read_csv(source/'selected_metadata.csv').numeric_stage.to_numpy(float)
    symbols = pd.read_csv(source/'genes.csv').symbol.fillna('').tolist()
    donors,donor_rows = read_cells(anchor_path,panel,1500,plan['donor_seed'])
    np.save(out/'anchor_rows.npy',donor_rows)
    previous_rows = np.load(old/'anchor_rows.npy')
    np.testing.assert_array_equal(donor_rows,previous_rows)
    saved = np.load(old/'encoder4096.npz')
    features = saved['features']
    net = DensityFlowNet(saved['basis'],saved['pca_center'],8.5,7.25)
    checkpoint = torch.load(old/'features4096.pt',weights_only=False,map_location='cpu')
    net.load_state_dict(checkpoint['net']);net.eval()
    program = FullAnchorSlopeForecast(x,stages,8.5,donors,panel,symbols,net,
                                      saved['center'],saved['scale'],features,np.load(old/'features.npy'),
                                      anchor_path=anchor_path)
    np.savez_compressed(out/'calibration.npz',positive_delta=program.anchor_positive_delta,
                        detection_delta=program.anchor_detection_delta,
                        positive_center=program.anchor_positive_center,latent_mean=program.anchor_latent_mean,
                        count=program.anchor_count,support=program.calibration_support)
    append_event(events,'all_E8.5_slopes_fitted',rows=program.audit['anchor_calibration_rows'],
                 calibration_sha256=digest(out/'calibration.npz'))
    generation = {}
    cache = TemporaryForecastCache(HERE/'private/temporary_cache')
    try:
        for name in candidates:
            if name=='copy':
                pred,indices,audit=donors.copy(),np.arange(len(donors)),{'method':'Measured E8.5 persistence'}
            elif name=='log1p_joint_a0.5':
                if digest(control)!=plan['archive_sha256'][str(control.relative_to(HERE))]:raise ValueError('Archived control changed')
                pred=np.load(control);indices=np.load(control.with_name('log1p_joint_a0.5_indices.npy'))
                audit={'method':'Archived matched 1,500-anchor control','source':str(control.relative_to(HERE))}
            else:
                positive,detection=(.25,.75) if name=='full_anchor_p0.25_d0.75' else (.5,.5)
                program.configure(positive,detection)
                pred,indices,audit=program.predict(9.5,'joint',1.,sampling='systematic')
            np.save(out/(name+'_indices.npy'),indices)
            generation[name]={'prediction_sha256':cache.put(name,pred),'audit':audit,
                              'retention':'temporary_D_cache_until_scored'}
            append_event(events,'forecast_frozen',candidate=name,prediction_sha256=generation[name]['prediction_sha256'])
            del pred
        (out/'generation.json').write_text(json.dumps(generation,indent=2))
        append_event(events,'all_forecasts_frozen_before_target_read')
        core,_=load_core()
        reference=json.loads((old/'report.json').read_text())
        report={'plan':plan,'generation':generation,'panels':[],
                'scorer_manifest_sha256':digest(HERE/'private/scorer_source/manifest.json'),
                'official_score':None,'local_gate_passed':False}
        for seed in seeds:
            future,rows=read_cells(target_path,panel,2000,seed)
            np.save(out/f'target_rows_{seed}.npy',rows)
            order=np.random.default_rng(seed).permutation(len(future))
            evaluator=Panel(core,future[order[:1000]],donors,seed)
            floor=evaluator.metrics(donors);ceiling=evaluator.metrics(future[order[1000:]])
            prior=next(p for p in reference['panels'] if p['seed']==seed)
            if floor!=prior['floor'] or ceiling!=prior['ceiling']:raise ValueError('Calibration mismatch')
            scores=[]
            for name in candidates:
                # Keep the D-only cache handle across panels; release each array.
                with cache.read(name,consume=False) as pred:
                    raw=evaluator.metrics(pred)
                result={'candidate':name,'prediction_sha256':generation[name]['prediction_sha256'],
                        'raw_metrics':raw,**evaluator.aggregate(raw,floor,ceiling)}
                scores.append(result)
                append_event(events,'candidate_scored',seed=seed,**result)
            report['panels'].append({'seed':seed,'cutoff':8.5,'target':9.5,
                                     'floor':floor,'ceiling':ceiling,'results':scores})
            (out/'report.partial.json').write_text(json.dumps(report,indent=2))
        report['status']='completed'
        (out/'report.json').write_text(json.dumps(report,indent=2))
        summaries=[]
        for name in candidates:
            results=[next(r for r in p['results'] if r['candidate']==name) for p in report['panels']]
            valid=all(r['calibration_valid'] for r in results)
            summaries.append({'candidate':name,'scores':[r['local_score'] for r in results],
                              'mean_score':float(np.mean([r['local_score'] for r in results])) if valid else None,
                              'raw_metrics':[r['raw_metrics'] for r in results],
                              'skills':[r['skills'] for r in results],'all_calibrations_valid':valid})
        public={'updated_utc':now(),'status':'completed_not_promoted','summaries':summaries,
                'panels':report['panels'],'plan_sha256':digest(out/'plan.json'),
                'report_sha256':digest(out/'report.json'),'scope':plan['scope'],
                'local_72_gate_passed':False,'official_score':None,'submissions_used':0}
        (HERE/'CNF_FULL_ANCHOR_RESULTS.json').write_text(json.dumps(public,indent=2))
        path=HERE/'LOCAL_OPTIMIZATION_STATE.json';state=json.loads(path.read_text())
        state.setdefault('cnf_full_anchor_job',{}).update(
            status='completed_not_promoted',run=out.name,evaluations=12,pending_evaluations=0,
            report_sha256=public['report_sha256'],results_report='CNF_FULL_ANCHOR_RESULTS.json')
        state['active_jobs']=[];state['local_process_running']=False
        path.write_text(json.dumps(state,indent=2))
        path=HERE/'METRIC_RESEARCH_QUEUE.json';queue=json.loads(path.read_text())
        for entry in queue['paths']:
            if entry.get('run')==out.name:
                entry.update(status='implemented_evaluated_not_promoted',results_report='CNF_FULL_ANCHOR_RESULTS.json')
        path.write_text(json.dumps(queue,indent=2))
        from index_scores import main as index_scores
        index_scores()
        from update_compact_checkpoint import refresh
        refresh('Full E8.5 anchor slope trial complete on three reused development panels; all raw metrics/skills indexed. Review paired gains before promotion; >72 and temporal gates remain unmet.')
        append_event(events,'full_anchor_trial_completed',evaluations=12)
    finally:
        cache.close()


if __name__=='__main__':
    torch.set_num_threads(2)
    try:
        with threadpool_limits(limits=2):main()
    except Exception as exc:
        folder=HERE/'private/cnf_full_anchor_challenge_01'
        if folder.exists():
            (folder/'failure.json').write_text(json.dumps({'status':'execution_failed',
                'error_type':type(exc).__name__,'error':str(exc)},indent=2))
            append_event(folder/'events.jsonl','execution_failed',error_type=type(exc).__name__,error=str(exc))
        raise
