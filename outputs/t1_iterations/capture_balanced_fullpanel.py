"""Scorer-aligned one-day source-slope ablation with matched frozen controls."""
import json
from collections import Counter
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event
from cnf_manifold_flow import DensityFlowNet
from partial_anchor_forecast import PartialAnchorForecast
from temporary_forecast_cache import TemporaryForecastCache
from offline_backtest import load_core, Panel
from lineage_residual_screen import lineage
from robust_population import covariance_change
from metric_critique_reward import assess, total_reward
from capture_lineage_trends import fit_capture_trends


def main():
    root = HERE.parents[1]
    source = HERE / 'private/associated_prepared_01'
    archive = HERE / 'private/cnf_hurdle_temporal_01'
    out = HERE / 'private/capture_balanced_fullpanel_01'
    if out.exists():
        raise ValueError('Preserve frozen capture-balanced run')
    panel = (root / 'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
    folds = [(8., 9.), (8.25, 9.25)]
    names = ['copy', 'anchor_unshrunk', 'slope_recent', 'slope_cellweighted', 'slope_capturemedian']
    plan = {
        'created_utc': now(), 'code_sha256': digest(HERE / 'capture_balanced_fullpanel.py'),
        'model_source_sha256': {f: digest(HERE / f) for f in
                                ['partial_anchor_forecast.py', 'log1p_positive_forecast.py',
                                 'anchor_slope_calibration.py', 'feature_panel_forecast.py', 'lineage_time_partial_pool_proxy.py', 'metric_critique_reward.py', 'lineage_residual_screen.py',
                                 'temporary_forecast_cache.py', 'offline_backtest.py', 'capture_lineage_trends.py']},
        'prepared_report_sha256': digest(source / 'report.json'),
        'panel_sha256': digest(root / 'outputs/t1_run/T1__val.genes.txt'),
        'archive_sha256': {f'{cutoff}/{name}': digest(archive / f'cutoff_{cutoff}' / name)
                           for cutoff, _ in folds for name in ['encoder.npz', 'training.pt', 'heads.npz']},
        'folds': folds, 'candidates': names, 'seed': 20260928,
        'donor_count': 1500, 'target_count': 2000, 'scored_truth_count': 1000,
        'fit': 'Past-only three-stage equal-stage lineage OLS. Capture-median means (>=10cells/capture-lineage, >=2eligiblecaptures/stage-lineage) compared with matched pooled cell means, same current lineage weights and one-day four-step delta. Insufficient capture support explicitly pooled fallback; missing lineage stage gives zero slope. Same fixed backoffs/logmass/mean/cov guards. Leave-one-capture-out is past-fit sensitivity only, never independent temporal validation. Future targets never fit or choose candidates.',
        'control': 'Same donors/targets, source cutoff, scorer and calibration: persistence and unshrunk .5/.5 anchor decoder.',
        'scope': 'Two reused source-cohort one-day folds, five predictions per fold, full unchanged 32285-gene metric vector and calibration. These folds are not challenge-domain validation.',
        'decision': 'Advance only if a slope improves headline versus persistence and incumbent on EACH fold and preserves all four skills versus incumbent. Separate reward uses paired mean skills; >72 temporal/64-replicate gate remains unmet.',
        'retention': 'D-only temporary forecast handles; no full prediction arrays retained.',
        'submissions_allowed': 0, 'jev_requests_allowed': 0,
    }
    plan['slope_artifacts_sha256']={str(c):digest(HERE/'private/lineage_time_partial_pool_proxy_01'/f'deltas_{c}.npz') for c,_ in folds}
    replay_path=HERE/'private/joint_detection_coupling_temporal_01/report.json'
    plan['replay_controls_report_sha256']=digest(replay_path)
    replay_report=json.loads(replay_path.read_text())
    prior_path=HERE/'SOURCE_SLOPE_BACKOFF_RESULTS.json'
    plan['prior_report_sha256']=digest(prior_path)
    prior_report=json.loads(prior_path.read_text())
    plan['reward_candidates']=['slope_cellweighted','slope_capturemedian']
    plan['capture_minimum_cells']=10;plan['capture_minimum_count']=2
    plan['shrinkage_backoffs']=[.25,.125,.0625,0.];plan['horizon_steps']=4.;plan['maximum_mean_perturbation']=.5;plan['covariance_guard']=.4
    out.mkdir(); (out/'executed_source.py').write_bytes((HERE/'capture_balanced_fullpanel.py').read_bytes())
    (out / 'plan.json').write_text(json.dumps(plan, indent=2))
    events = out / 'events.jsonl'; append_event(events, 'plan_frozen', sha256=digest(out / 'plan.json'))
    state_path = HERE / 'LOCAL_OPTIMIZATION_STATE.json'; state = json.loads(state_path.read_text())
    state['active_jobs'] = [out.name]; state['local_process_running'] = True
    state['active_run_path'] = f'private/{out.name}'
    state['capture_balanced_fullpanel_job'] = {'status': 'running', 'planned_scores': 10,
                                     'completed_scores': 0, 'plan_sha256': digest(out / 'plan.json')}
    state_path.write_text(json.dumps(state, indent=2))
    prepared = json.loads((source / 'report.json').read_text())
    for name, key in [('expression.npy', 'expression_sha256'),
                      ('selected_metadata.csv', 'metadata_sha256'), ('genes.csv', 'genes_sha256')]:
        if digest(source / name) != prepared[key]:
            raise ValueError('Prepared input changed: ' + name)
    x = np.load(source / 'expression.npy', mmap_mode='r')
    metadata=pd.read_csv(source / 'selected_metadata.csv')
    stages = metadata.numeric_stage.to_numpy(float)
    labels=metadata.celltype_extended_atlas.map(lineage).to_numpy()
    metadata['lineage']=labels
    symbols = pd.read_csv(source / 'genes.csv').symbol.fillna('').tolist()
    counts = Counter(symbols)
    lookup = {s: i for i, s in enumerate(symbols) if s and counts[s] == 1}
    official = np.array([i for i, s in enumerate(panel) if s in lookup])
    atlas = np.array([lookup[panel[i]] for i in official])

    def values(rows):
        result = np.zeros((len(rows), len(panel)), dtype=np.float32)
        result[:, official] = np.asarray(x[np.ix_(rows, atlas)])
        return result

    core, _ = load_core()
    report = {'plan': plan, 'folds': [], 'status': 'running',
              'scorer_manifest_sha256': digest(HERE / 'private/scorer_source/manifest.json'),
              'official_score': None, 'local_gate_passed': False}
    for cutoff, target in folds:
        folder = out / f'cutoff_{cutoff}'; folder.mkdir()
        donor_rows = np.sort(np.random.default_rng(plan['seed']).choice(
            np.flatnonzero(stages == cutoff), plan['donor_count'], replace=False))
        np.save(folder / 'donor_rows.npy', donor_rows)
        donors = values(donor_rows)
        arc = archive / f'cutoff_{cutoff}'
        for name in ['encoder.npz', 'training.pt', 'heads.npz']:
            if digest(arc / name) != plan['archive_sha256'][f'{cutoff}/{name}']:
                raise ValueError('Archived past model changed: ' + name)
        with np.load(arc / 'encoder.npz') as encoded:
            features = encoded['features']; guard = encoded['guard_features']
            center = encoded['center']; scale = encoded['scale']
            net = DensityFlowNet(encoded['basis'], encoded['pca_center'], cutoff, 7.25)
        net.load_state_dict(torch.load(arc / 'training.pt', weights_only=False, map_location='cpu')['net'])
        net.eval()
        program = PartialAnchorForecast(x, stages, cutoff, donors, panel, symbols, net,
                                        center, scale, features, guard)
        program.configure(.5, .5)
        slope_path=HERE/'private/lineage_time_partial_pool_proxy_01'/f'deltas_{cutoff}.npz'
        if digest(slope_path)!=plan['slope_artifacts_sha256'][str(cutoff)]:raise ValueError('Past slope artifact changed')
        with np.load(slope_path) as saved:
            if not np.array_equal(saved['mapped'],official):raise ValueError('Slope mapping mismatch')
            slopes={'slope_recent':saved['recent_linear'].copy()}
        new_slopes,fit_audit=fit_capture_trends(x,metadata,atlas,cutoff)
        slopes.update(new_slopes)
        slope_bank=folder/'capture_slopes.npz';np.savez_compressed(slope_bank,mapped=official,**new_slopes)
        (folder/'capture_fit_audit.json').write_text(json.dumps(fit_audit,indent=2))
        append_event(events,'capture_trends_frozen_before_target',cutoff=cutoff,slopes_sha256=digest(slope_bank),fit_audit_sha256=digest(folder/'capture_fit_audit.json'))
        append_event(events,'past_only_slope_artifacts_loaded',cutoff=cutoff,sha256=digest(slope_path))
        cache = TemporaryForecastCache(HERE / 'private/temporary_cache')
        generation = {}
        try:
            for name in names:
                if name == 'copy':
                    pred, ids, audit = donors.copy(), np.arange(len(donors)), {'method': 'persistence'}
                elif name=='anchor_unshrunk':
                    pred,ids,audit=program.predict(target,'joint',1.,sampling='systematic')
                else:
                    ids=np.arange(len(donors));attempts=[]
                    original_mass=donors[:,official].sum(1,dtype=np.float64)
                    for alpha in plan['shrinkage_backoffs']:
                        pred=donors.copy()
                        if alpha:
                            pred[:,official]=np.maximum(0,pred[:,official]+plan['horizon_steps']*alpha*slopes[name])
                            changed_mass=pred[:,official].sum(1,dtype=np.float64)
                            factor=np.divide(original_mass,changed_mass,out=np.ones_like(original_mass),where=changed_mass>0)
                            pred[:,official]*=factor[:,None]
                        mean_change=float(np.max(np.abs(pred.mean(0,dtype=np.float64)-donors.mean(0,dtype=np.float64))))
                        change=covariance_change(donors[:,guard],pred[:,guard]) if alpha else 0.
                        mass_error=float(np.max(np.abs(pred[:,official].sum(1,dtype=np.float64)-original_mass)/np.maximum(original_mass,1e-9)))
                        valid=bool(np.isfinite(pred).all() and np.all(pred>=0) and mass_error<=1e-5)
                        rejected=not valid or not np.isfinite(change) or change>plan['covariance_guard'] or mean_change>plan['maximum_mean_perturbation']
                        attempt={'alpha':alpha,'covariance_change':change,'maximum_mean_perturbation':mean_change,'mapped_mass_relative_error':mass_error,'valid':valid,'guard_rejected':bool(rejected)}
                        attempts.append(attempt)
                        append_event(events,'past_only_backoff_attempt',cutoff=cutoff,candidate=name,**attempt)
                        if not rejected:break
                        del pred
                    else:raise ValueError('No valid backoff including persistence')
                    audit={'method':'anchored_past_slope_backoff','slope_artifact_sha256':digest(slope_path),'horizon_steps':4,'selected_shrinkage':alpha,'attempts':attempts,'persistence_guard_fallback':alpha==0}
                np.save(folder / f'{name}_indices.npy', ids)
                generation[name] = {'prediction_sha256': cache.put(name, pred), 'audit': audit}
                if name in ['copy','anchor_unshrunk']:
                    old=next(f for f in replay_report['folds'] if f['cutoff']==cutoff)
                    if generation[name]['prediction_sha256']!=old['generation'][name]['prediction_sha256']:raise ValueError('Frozen control forecast replay mismatch')
                if name=='slope_recent':
                    old=next(f for f in prior_report['folds'] if f['cutoff']==cutoff)
                    if generation[name]['prediction_sha256']!=old['generation'][name]['prediction_sha256']:raise ValueError('Recent control forecast replay failed')
                append_event(events, 'forecast_frozen', cutoff=cutoff, candidate=name,
                             prediction_sha256=generation[name]['prediction_sha256'])
                del pred
            (folder / 'generation.json').write_text(json.dumps(generation, indent=2))
            append_event(events, 'all_forecasts_frozen_before_target_read', cutoff=cutoff)
            target_rows = np.sort(np.random.default_rng(plan['seed']).choice(
                np.flatnonzero(stages == target), plan['target_count'], replace=False))
            np.save(folder / 'target_rows.npy', target_rows)
            old=next(f for f in prior_report['folds'] if f['cutoff']==cutoff)
            if digest(folder/'donor_rows.npy')!=old['donor_rows_sha256'] or digest(folder/'target_rows.npy')!=old['target_rows_sha256']:raise ValueError('Matched split replay failed')
            future = values(target_rows)
            order = np.random.default_rng(plan['seed']).permutation(len(future))
            evaluator = Panel(core, future[order[:1000]], donors, plan['seed'])
            floor = evaluator.metrics(donors)
            ceiling = evaluator.metrics(future[order[1000:]])
            rows = []
            raw_by_hash={generation['copy']['prediction_sha256']:floor}
            for name in names:
                key=generation[name]['prediction_sha256']
                reused_metrics=key in raw_by_hash
                with cache.read(name, consume=True) as pred:
                    if reused_metrics:raw=raw_by_hash[key]
                    else:
                        raw=evaluator.metrics(pred);raw_by_hash[key]=raw
                append_event(events,'metric_cache_used',cutoff=cutoff,candidate=name,reused=reused_metrics,prediction_sha256=key)
                row = {'candidate': name, 'prediction_sha256': generation[name]['prediction_sha256'],
                       'raw_metrics': raw, **evaluator.aggregate(raw, floor, ceiling)}
                rows.append(row)
                append_event(events, 'candidate_scored', cutoff=cutoff, target=target, **row)
                state = json.loads(state_path.read_text())
                state['capture_balanced_fullpanel_job']['completed_scores'] += 1
                state_path.write_text(json.dumps(state, indent=2))
            report['folds'].append({'cutoff': cutoff, 'target': target, 'floor': floor,
                                    'ceiling': ceiling, 'results': rows, 'generation': generation, 'capture_fit_audit':fit_audit,
                                    'donor_rows_sha256': digest(folder / 'donor_rows.npy'),
                                    'target_rows_sha256': digest(folder / 'target_rows.npy')})
            (out / 'report.partial.json').write_text(json.dumps(report, indent=2))
        finally:
            cache.close()
        del program, net, donors, future, evaluator
    assessments=[]
    for name in plan['reward_candidates']:
        candidate=[next(r for r in f['results'] if r['candidate']==name) for f in report['folds']]
        incumbent=[next(r for r in f['results'] if r['candidate']=='anchor_unshrunk') for f in report['folds']]
        eligible=all(r['calibration_valid'] for r in candidate+incumbent)
        assessment=assess([r['skills'] for r in candidate],[r['skills'] for r in incumbent],eligible=eligible)
        assessments.append({'experiment_id':out.name+'/'+name,'assessment':assessment,'plan_sha256':digest(out/'plan.json')})
    report['critic_assessments']=assessments
    report['status'] = 'completed'
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    summary = []
    for name in names:
        rows = [next(r for r in f['results'] if r['candidate'] == name) for f in report['folds']]
        summary.append({'candidate': name, 'scores': [r['local_score'] for r in rows],
                        'raw_metrics': [r['raw_metrics'] for r in rows],
                        'skills': [r['skills'] for r in rows],
                        'all_calibrations_valid': all(r['calibration_valid'] for r in rows)})
    controls = {x['candidate']: x for x in summary}
    passed = [x['candidate'] for x in summary if x['candidate'].startswith('slope_') and
              x['all_calibrations_valid'] and all(
                  x['scores'][i] > max(controls['copy']['scores'][i],
                                       controls['anchor_unshrunk']['scores'][i]) and
                  all(x['skills'][i][m] >= controls['anchor_unshrunk']['skills'][i][m]
                      for m in ['de_score', 'de_direction', 'mmd_u', 'variogram'])
                  for i in range(len(folds)))]
    public = {'updated_utc': now(), 'status': 'completed', 'summary': summary,
              'folds': report['folds'], 'passing_candidates': passed,
              'plan_sha256': digest(out / 'plan.json'),
              'report_sha256': digest(out / 'report.json'),
              'official_score': None, 'local_72_gate_passed': False,
              'submissions_used': 0}
    (HERE / 'CAPTURE_BALANCED_FULLPANEL_RESULTS.json').write_text(json.dumps(public, indent=2))
    reward_path=HERE/'METRIC_CRITIQUE_REWARD_LEDGER.jsonl'
    previous=[json.loads(line) for line in reward_path.read_text().splitlines() if line.strip()]
    known={e['experiment_id'] for e in previous}
    additions=[e for e in assessments if e['experiment_id'] not in known]
    for e in additions:e['report_sha256']=public['report_sha256']
    with reward_path.open('a') as handle:
        for event in additions:handle.write(json.dumps(event)+'\n')
    totals=total_reward(previous+additions)
    state = json.loads(state_path.read_text()); state['active_jobs'] = []
    state['metric_critique_reward'].update(current_reward=totals['reward_score'],uncapped_reward=totals['uncapped_reward'])
    state['local_process_running'] = False
    state['active_run_path']=None
    state['capture_balanced_fullpanel_job'].update(status='completed', report_sha256=public['report_sha256'],
                                          passing_candidates=passed)
    state_path.write_text(json.dumps(state, indent=2))
    append_event(events, 'temporal_batch_completed', passing_candidates=passed,
                 full_panel_scores=len(folds)*len(names))
    from index_scores import main as index_scores
    index_scores()


if __name__ == '__main__':
    torch.set_num_threads(2)
    try:
        with threadpool_limits(limits=2):main()
    except Exception as exc:
        failed=HERE/'private/capture_balanced_fullpanel_01'
        if failed.exists():
            append_event(failed/'events.jsonl','run_failed',exception_type=type(exc).__name__,message=str(exc))
            sp=HERE/'LOCAL_OPTIMIZATION_STATE.json';state=json.loads(sp.read_text())
            state['active_jobs']=[];state['local_process_running']=False;state['active_run_path']=None
            state.setdefault('capture_balanced_fullpanel_job',{}).update(status='failed',exception_type=type(exc).__name__,message=str(exc))
            sp.write_text(json.dumps(state,indent=2))
        raise
