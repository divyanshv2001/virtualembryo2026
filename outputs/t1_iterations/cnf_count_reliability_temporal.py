"""Count-aware anchor-slope shrinkage on two past-only source temporal folds."""
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


def main():
    root = HERE.parents[1]
    source = HERE / 'private/associated_prepared_01'
    archive = HERE / 'private/cnf_hurdle_temporal_01'
    out = HERE / 'private/cnf_count_reliability_temporal_01'
    if out.exists():
        raise ValueError('Preserve frozen count-reliability run')
    panel = (root / 'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
    folds = [(8., 9.), (8.25, 9.25)]
    names = ['copy', 'anchor_unshrunk', 'anchor_count_k100', 'anchor_count_k500']
    plan = {
        'created_utc': now(), 'code_sha256': digest(HERE / 'cnf_count_reliability_temporal.py'),
        'model_source_sha256': {f: digest(HERE / f) for f in
                                ['partial_anchor_forecast.py', 'log1p_positive_forecast.py',
                                 'anchor_slope_calibration.py', 'feature_panel_forecast.py',
                                 'temporary_forecast_cache.py', 'offline_backtest.py']},
        'prepared_report_sha256': digest(source / 'report.json'),
        'panel_sha256': digest(root / 'outputs/t1_run/T1__val.genes.txt'),
        'archive_sha256': {f'{cutoff}/{name}': digest(archive / f'cutoff_{cutoff}' / name)
                           for cutoff, _ in folds for name in ['encoder.npz', 'training.pt', 'heads.npz']},
        'folds': folds, 'candidates': names, 'seed': 20260928,
        'donor_count': 1500, 'target_count': 2000, 'scored_truth_count': 1000,
        'fit': 'Frozen past-only encoder/800-update CNF at each source cutoff. Refit log1p conditional source heads and same-stage 1500-donor anchor slopes, then multiply positive and detection anchor delta for each gene by count/(count+k). k=0,100,500; original .5/.5 strengths and all other dynamics/guards unchanged.',
        'hypothesis': 'Low-positive-count anchor gene slopes are unstable; empirical count shrinkage may improve temporal transfer of DE, direction, MMD or variogram without suppressing well-supported genes.',
        'control': 'Same donors/targets, source cutoff, scorer and calibration: persistence and unshrunk .5/.5 anchor decoder.',
        'scope': 'Two reused source-cohort one-day folds, four predictions per fold, full unchanged 32285-gene metric vector and calibration. These folds are not challenge-domain validation.',
        'decision': 'Advance only if one k>0 improves headline versus both controls on each fold with all four metric skills preserved; otherwise reject this bounded attenuation. Final >72 mean/lower-tail 64-replicate and temporal readiness gates remain required.',
        'retention': 'D-only temporary forecast handles; no full prediction arrays retained.',
        'submissions_allowed': 0, 'jev_requests_allowed': 0,
    }
    out.mkdir(); (out / 'plan.json').write_text(json.dumps(plan, indent=2))
    events = out / 'events.jsonl'; append_event(events, 'plan_frozen', sha256=digest(out / 'plan.json'))
    state_path = HERE / 'LOCAL_OPTIMIZATION_STATE.json'; state = json.loads(state_path.read_text())
    state['active_jobs'] = [out.name]; state['local_process_running'] = True
    state['active_run_path'] = f'private/{out.name}'
    state['count_reliability_job'] = {'status': 'running', 'planned_scores': 8,
                                     'completed_scores': 0, 'plan_sha256': digest(out / 'plan.json')}
    state_path.write_text(json.dumps(state, indent=2))
    prepared = json.loads((source / 'report.json').read_text())
    for name, key in [('expression.npy', 'expression_sha256'),
                      ('selected_metadata.csv', 'metadata_sha256'), ('genes.csv', 'genes_sha256')]:
        if digest(source / name) != prepared[key]:
            raise ValueError('Prepared input changed: ' + name)
    x = np.load(source / 'expression.npy', mmap_mode='r')
    stages = pd.read_csv(source / 'selected_metadata.csv').numeric_stage.to_numpy(float)
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
        original_positive = program.anchor_positive_delta.copy()
        original_detection = program.anchor_detection_delta.copy()
        anchor_count = program.anchor_count.copy()
        np.savez_compressed(folder / 'fit.npz', anchor_count=anchor_count,
                            support=program.calibration_support,
                            positive_delta=original_positive,
                            detection_delta=original_detection)
        append_event(events, 'past_only_heads_fitted', cutoff=cutoff,
                     fit_sha256=digest(folder / 'fit.npz'),
                     supported_genes=int(program.calibration_support.sum()))
        cache = TemporaryForecastCache(HERE / 'private/temporary_cache')
        generation = {}
        try:
            for name in names:
                if name == 'copy':
                    pred, ids, audit = donors.copy(), np.arange(len(donors)), {'method': 'persistence'}
                else:
                    k = 0 if name == 'anchor_unshrunk' else int(name.rsplit('k', 1)[1])
                    weight = anchor_count / (anchor_count + k) if k else np.ones_like(anchor_count, dtype=float)
                    program.anchor_positive_delta = original_positive * weight[None, :]
                    program.anchor_detection_delta = original_detection * weight[None, :]
                    pred, ids, audit = program.predict(target, 'joint', 1., sampling='systematic')
                    audit = dict(audit, count_shrink_k=k)
                np.save(folder / f'{name}_indices.npy', ids)
                generation[name] = {'prediction_sha256': cache.put(name, pred), 'audit': audit}
                append_event(events, 'forecast_frozen', cutoff=cutoff, candidate=name,
                             prediction_sha256=generation[name]['prediction_sha256'])
                del pred
            (folder / 'generation.json').write_text(json.dumps(generation, indent=2))
            append_event(events, 'all_forecasts_frozen_before_target_read', cutoff=cutoff)
            target_rows = np.sort(np.random.default_rng(plan['seed']).choice(
                np.flatnonzero(stages == target), plan['target_count'], replace=False))
            np.save(folder / 'target_rows.npy', target_rows)
            future = values(target_rows)
            order = np.random.default_rng(plan['seed']).permutation(len(future))
            evaluator = Panel(core, future[order[:1000]], donors, plan['seed'])
            floor = evaluator.metrics(donors)
            ceiling = evaluator.metrics(future[order[1000:]])
            rows = []
            for name in names:
                with cache.read(name, consume=True) as pred:
                    raw = evaluator.metrics(pred)
                row = {'candidate': name, 'prediction_sha256': generation[name]['prediction_sha256'],
                       'raw_metrics': raw, **evaluator.aggregate(raw, floor, ceiling)}
                rows.append(row)
                append_event(events, 'candidate_scored', cutoff=cutoff, target=target, **row)
                state = json.loads(state_path.read_text())
                state['count_reliability_job']['completed_scores'] += 1
                state_path.write_text(json.dumps(state, indent=2))
            report['folds'].append({'cutoff': cutoff, 'target': target, 'floor': floor,
                                    'ceiling': ceiling, 'results': rows, 'generation': generation,
                                    'donor_rows_sha256': digest(folder / 'donor_rows.npy'),
                                    'target_rows_sha256': digest(folder / 'target_rows.npy')})
            (out / 'report.partial.json').write_text(json.dumps(report, indent=2))
        finally:
            cache.close()
        del program, net, donors, future, evaluator
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
    passed = [x['candidate'] for x in summary if x['candidate'].startswith('anchor_count_') and
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
    (HERE / 'CNF_COUNT_RELIABILITY_TEMPORAL_RESULTS.json').write_text(json.dumps(public, indent=2))
    state = json.loads(state_path.read_text()); state['active_jobs'] = []
    state['local_process_running'] = False
    state['count_reliability_job'].update(status='completed', report_sha256=public['report_sha256'],
                                          passing_candidates=passed)
    state_path.write_text(json.dumps(state, indent=2))
    append_event(events, 'temporal_batch_completed', passing_candidates=passed,
                 full_panel_scores=len(folds)*len(names))
    from index_scores import main as index_scores
    index_scores()


if __name__ == '__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        main()
