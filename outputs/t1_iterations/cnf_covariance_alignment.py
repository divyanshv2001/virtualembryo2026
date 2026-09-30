"""Frozen past-only 8D covariance alignment with full-panel matched controls."""
import json
from collections import Counter

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from threadpoolctl import threadpool_limits

from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event
from hurdle_backtest import read_cells
from cnf_manifold_flow import DensityFlowNet
from cnf_domain_mean_alignment import ShiftedTrajectory
from full_anchor_slope_forecast import FullAnchorSlopeForecast
from temporary_forecast_cache import TemporaryForecastCache
from offline_backtest import load_core, Panel


def symmetric_power(matrix, power):
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    return (vectors * np.maximum(values, 1e-8)**power) @ vectors.T


class CovarianceAlignedTrajectory:
    def __init__(self, field, source_mean, challenge_mean, transform):
        self.field = field
        self.source_mean = torch.tensor(source_mean, dtype=torch.float32)
        self.challenge_mean = torch.tensor(challenge_mean, dtype=torch.float32)
        self.transform = torch.tensor(transform, dtype=torch.float32)
        self.inverse = torch.tensor(np.linalg.inv(transform), dtype=torch.float32)

    def encode(self, values):
        return self.field.encode(values)

    def trajectory(self, z, times, step=.125):
        source_z = (z - self.challenge_mean) @ self.transform.T + self.source_mean
        source_path = self.field.trajectory(source_z, times, step=step)
        return (source_path - self.source_mean) @ self.inverse.T + self.challenge_mean


def main():
    root = HERE.parents[1]
    out = HERE / 'private/cnf_covariance_alignment_01'
    if out.exists():
        raise ValueError('Preserve frozen covariance run; inspect it instead of overwriting')
    prepared = HERE / 'private/associated_prepared_01'
    archived = HERE / 'private/cnf_feature_challenge_01'
    previous = HERE / 'private/cnf_domain_mean_alignment_01'
    anchor_path = root / 'data/E8.5_RNA.h5ad'
    target_path = root / 'data/E9.5_RNA.h5ad'
    panel_path = root / 'outputs/t1_run/T1__val.genes.txt'
    panel = panel_path.read_text().splitlines()
    seeds = [20260928, 20260929, 20260930]
    names = ['copy', 'mean_shift', 'covariance_0.25', 'covariance_0.5', 'covariance_1.0']
    plan = {
        'created_utc': now(),
        'source_sha256': {f: digest(HERE / f) for f in
                          ['cnf_covariance_alignment.py', 'cnf_domain_mean_alignment.py',
                           'full_anchor_slope_forecast.py', 'partial_anchor_forecast.py',
                           'log1p_positive_forecast.py', 'feature_panel_forecast.py',
                           'offline_backtest.py', 'temporary_forecast_cache.py']},
        'input_sha256': {str(p.relative_to(root)): digest(p) for p in
                         [anchor_path, target_path, panel_path]},
        'prepared_report_sha256': digest(prepared / 'report.json'),
        'encoder_sha256': digest(archived / 'encoder4096.npz'),
        'flow_sha256': digest(archived / 'features4096.pt'),
        'previous_report_sha256': digest(previous / 'report.json'),
        'previous_generation_sha256': digest(previous / 'generation.json'),
        'previous_alignment_sha256': digest(previous / 'alignment.npz'),
        'cutoff': 8.5, 'target': 9.5, 'donor_seed': 20260928,
        'scoring_seeds': seeds, 'donors': 1500, 'targets_per_seed': 2000,
        'candidate_names': names, 'covariance_alpha': [.25, .5, 1.],
        'covariance_ridge_fraction': .05,
        'full_map_eigenvalue_clip': [.5, 2.],
        'mechanism': 'Fit source/challenge E8.5 latent means and 8D covariances only. A symmetric positive definite Gaussian optimal-transport map sends challenge covariance toward source covariance; eigenvalues are clipped to [.5,2]. Interpolate I to this map at alpha .25/.5/1, transform E8.5 donors into source frame, integrate frozen source CNF, invert map. Decoder and .25/.75 full-anchor slopes remain fixed.',
        'controls': 'Identical E8.5 donors, E9.5 target panels, scorer and calibration: persistence and exact SHA256 replay of previously frozen full mean shift.',
        'scope': 'Three reused challenge development panels, 15 unchanged 32285-gene scores. Source/challenge E8.5 covariance can reflect tissue composition and assay, not causal biology; no E9.5 fit.',
        'decision_rule': 'Consider a covariance variant only if it improves all three panels over both controls, has no mean four-skill regression versus mean shift, and valid calibration. No 16-resample expansion unless point mean>=60; final >72 and temporal gates unchanged.',
        'storage': 'D-only temporary forecasts; retain plans, hashes, small indices, all raw metrics/skills/calibration and genuine events. No full prediction matrices retained.',
        'submissions_allowed': 0, 'jev_requests_allowed': 0,
    }
    out.mkdir()
    (out / 'plan.json').write_text(json.dumps(plan, indent=2))
    events = out / 'events.jsonl'
    append_event(events, 'plan_frozen', sha256=digest(out / 'plan.json'))
    state_path = HERE / 'LOCAL_OPTIMIZATION_STATE.json'
    state = json.loads(state_path.read_text())
    state['active_jobs'] = [out.name]
    state['active_run_path'] = f'private/{out.name}'
    state['local_process_running'] = True
    state['covariance_alignment_job'] = {'status': 'running', 'planned_scores': 15,
                                         'completed_scores': 0, 'plan_sha256': digest(out / 'plan.json')}
    state_path.write_text(json.dumps(state, indent=2))
    prepared_report = json.loads((prepared / 'report.json').read_text())
    for name, key in [('expression.npy', 'expression_sha256'),
                      ('selected_metadata.csv', 'metadata_sha256'), ('genes.csv', 'genes_sha256')]:
        if digest(prepared / name) != prepared_report[key]:
            raise ValueError('Prepared source changed: ' + name)
    x = np.load(prepared / 'expression.npy', mmap_mode='r')
    stages = pd.read_csv(prepared / 'selected_metadata.csv').numeric_stage.to_numpy(float)
    symbols = pd.read_csv(prepared / 'genes.csv').symbol.fillna('').tolist()
    donors, donor_rows = read_cells(anchor_path, panel, 1500, plan['donor_seed'])
    np.testing.assert_array_equal(donor_rows, np.load(previous / 'donor_rows.npy'))
    np.save(out / 'donor_rows.npy', donor_rows)
    with np.load(archived / 'encoder4096.npz') as encoder:
        features = encoder['features']; center = encoder['center']; scale = encoder['scale']
        net = DensityFlowNet(encoder['basis'], encoder['pca_center'], 8.5, 7.25)
    net.load_state_dict(torch.load(archived / 'features4096.pt',
                                   weights_only=False, map_location='cpu')['net'])
    net.eval()
    program = FullAnchorSlopeForecast(x, stages, 8.5, donors, panel, symbols, net,
                                      center, scale, features, np.load(archived / 'features.npy'),
                                      anchor_path=anchor_path)
    program.configure(.25, .75)
    counts = Counter(symbols)
    lookup = {s: i for i, s in enumerate(symbols) if s and counts[s] == 1}
    atlas_features = np.array([lookup[panel[i]] for i in features])
    source_rows = np.flatnonzero(stages == 8.5)
    with torch.no_grad():
        source_z = net.encode(torch.tensor(
            (np.asarray(x[np.ix_(source_rows, atlas_features)]) - center) / scale,
            dtype=torch.float32))[0].numpy().astype(np.float64)
    challenge_z = np.empty((program.audit['anchor_calibration_rows'], len(program.zcenter)), dtype=np.float64)
    a = ad.read_h5ad(anchor_path, backed='r')
    try:
        with torch.no_grad():
            for start in range(0, len(challenge_z), 256):
                end = min(start + 256, len(challenge_z))
                block = a.X[start:end, features]
                block = block.toarray() if sparse.issparse(block) else np.asarray(block)
                challenge_z[start:end] = net.encode(torch.tensor(
                    (block.astype(np.float32) - center) / scale))[0].numpy()
    finally:
        a.file.close()
    with np.load(previous / 'alignment.npz') as saved:
        source_mean = saved['source_mean']; challenge_mean = saved['challenge_mean']
        delta = saved['delta']
    np.testing.assert_allclose(source_z.mean(0), source_mean, atol=1e-6)
    np.testing.assert_allclose(challenge_z.mean(0), challenge_mean, atol=1e-6)
    cs = np.cov(source_z, rowvar=False); cc = np.cov(challenge_z, rowvar=False)
    ridge = .05 * (np.trace(cs) + np.trace(cc)) / (2 * cs.shape[0])
    cs += ridge * np.eye(cs.shape[0]); cc += ridge * np.eye(cc.shape[0])
    root_c = symmetric_power(cc, .5)
    inv_root_c = symmetric_power(cc, -.5)
    target = inv_root_c @ symmetric_power(root_c @ cs @ root_c, .5) @ inv_root_c
    eig, vectors = np.linalg.eigh((target + target.T) / 2)
    clipped = np.clip(eig, .5, 2.)
    target = (vectors * clipped) @ vectors.T
    np.savez_compressed(out / 'alignment.npz', source_mean=source_mean,
                        challenge_mean=challenge_mean, source_cov=cs, challenge_cov=cc,
                        full_map=target, full_map_eigenvalues=eig, clipped_eigenvalues=clipped)
    append_event(events, 'past_only_covariance_map_fitted', source_rows=len(source_rows),
                 challenge_rows=len(challenge_z), ridge=float(ridge),
                 eigenvalues=eig.tolist(), clipped_eigenvalues=clipped.tolist(),
                 alignment_sha256=digest(out / 'alignment.npz'))
    expected = json.loads((previous / 'generation.json').read_text())
    cache = TemporaryForecastCache(HERE / 'private/temporary_cache')
    try:
        generation = {}
        for name in names:
            if name == 'copy':
                pred, ids, audit = donors.copy(), np.arange(len(donors)), {'method': 'persistence'}
            else:
                if name == 'mean_shift':
                    program.net = ShiftedTrajectory(net, delta)
                else:
                    alpha = float(name.split('_')[1])
                    transform = np.eye(len(delta)) + alpha * (target - np.eye(len(delta)))
                    program.net = CovarianceAlignedTrajectory(net, source_mean,
                                                               challenge_mean, transform)
                pred, ids, audit = program.predict(9.5, 'joint', 1., sampling='systematic')
                audit = dict(audit, covariance_alignment_strength=0. if name == 'mean_shift' else alpha)
            np.save(out / f'{name}_indices.npy', ids)
            sha = cache.put(name, pred)
            if name in ('copy', 'mean_shift'):
                previous_name = 'copy' if name == 'copy' else 'shift_full'
                if sha != expected[previous_name]['prediction_sha256']:
                    raise ValueError('Frozen control hash mismatch: ' + name)
            generation[name] = {'prediction_sha256': sha, 'audit': audit}
            append_event(events, 'forecast_frozen', candidate=name, prediction_sha256=sha)
            del pred
        (out / 'generation.json').write_text(json.dumps(generation, indent=2))
        append_event(events, 'all_forecasts_frozen_before_target_read')
        core, _ = load_core()
        previous_report = json.loads((previous / 'report.json').read_text())
        report = {'status': 'running', 'plan': plan, 'generation': generation, 'panels': [],
                  'scorer_manifest_sha256': digest(HERE / 'private/scorer_source/manifest.json'),
                  'official_score': None, 'local_gate_passed': False}
        for seed in seeds:
            future, rows = read_cells(target_path, panel, 2000, seed)
            np.save(out / f'target_rows_{seed}.npy', rows)
            order = np.random.default_rng(seed).permutation(len(future))
            evaluator = Panel(core, future[order[:1000]], donors, seed)
            floor = evaluator.metrics(donors)
            ceiling = evaluator.metrics(future[order[1000:]])
            old = next(p for p in previous_report['panels'] if p['seed'] == seed)
            if floor != old['floor'] or ceiling != old['ceiling']:
                raise ValueError('Calibration panel changed')
            outcomes = []
            for name in names:
                with cache.read(name, consume=False) as pred:
                    raw = evaluator.metrics(pred)
                row = {'candidate': name, 'prediction_sha256': generation[name]['prediction_sha256'],
                       'raw_metrics': raw, **evaluator.aggregate(raw, floor, ceiling)}
                outcomes.append(row)
                append_event(events, 'candidate_scored', seed=seed, **row)
                state = json.loads(state_path.read_text())
                state['covariance_alignment_job']['completed_scores'] += 1
                state_path.write_text(json.dumps(state, indent=2))
            report['panels'].append({'seed': seed, 'cutoff': 8.5, 'target': 9.5,
                                     'floor': floor, 'ceiling': ceiling, 'results': outcomes,
                                     'target_rows_sha256': digest(out / f'target_rows_{seed}.npy')})
            (out / 'report.partial.json').write_text(json.dumps(report, indent=2))
        report['status'] = 'completed'
        (out / 'report.json').write_text(json.dumps(report, indent=2))
        summaries = []
        for name in names:
            outcomes = [next(r for r in p['results'] if r['candidate'] == name) for p in report['panels']]
            summaries.append({'candidate': name, 'scores': [r['local_score'] for r in outcomes],
                              'mean_score': float(np.mean([r['local_score'] for r in outcomes])),
                              'all_calibrations_valid': all(r['calibration_valid'] for r in outcomes),
                              'mean_skills': {m: float(np.mean([r['skills'][m] for r in outcomes]))
                                              for m in outcomes[0]['skills']}})
        by_name = {row['candidate']: row for row in summaries}
        passing = [name for name in names[2:] if by_name[name]['all_calibrations_valid'] and
                   all(by_name[name]['scores'][i] > max(by_name['copy']['scores'][i],
                                                       by_name['mean_shift']['scores'][i])
                       for i in range(3)) and
                   all(by_name[name]['mean_skills'][m] >= by_name['mean_shift']['mean_skills'][m]
                       for m in by_name[name]['mean_skills'])]
        public = {'updated_utc': now(), 'status': 'completed', 'summaries': summaries,
                  'panels': report['panels'], 'passing_candidates': passing,
                  'expand_to_16_resamples': any(by_name[n]['mean_score'] >= 60 for n in passing),
                  'plan_sha256': digest(out / 'plan.json'), 'report_sha256': digest(out / 'report.json'),
                  'official_score': None, 'local_72_gate_passed': False, 'submissions_used': 0}
        (HERE / 'CNF_COVARIANCE_ALIGNMENT_RESULTS.json').write_text(json.dumps(public, indent=2))
        state = json.loads(state_path.read_text())
        state['active_jobs'] = []; state['local_process_running'] = False; state['active_run_path'] = None
        state['covariance_alignment_job'].update(status='completed',
             report_sha256=public['report_sha256'], passing_candidates=passing,
             expand_to_16_resamples=public['expand_to_16_resamples'])
        state_path.write_text(json.dumps(state, indent=2))
        append_event(events, 'batch_completed', scores=15, passing_candidates=passing)
        from index_scores import main as index_scores
        index_scores()
    finally:
        cache.close()


if __name__ == '__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        main()
