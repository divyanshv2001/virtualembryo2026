"""Bounded past-only source/challenge mean-aligned CNF forecast trial."""
import json
from collections import Counter
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


class ShiftedTrajectory:
    def __init__(self, field, shift):
        self.field = field
        self.shift = torch.tensor(shift, dtype=torch.float32)

    def encode(self, values):
        return self.field.encode(values)

    def trajectory(self, z, times, step=.125):
        return self.field.trajectory(z-self.shift, times, step=step)+self.shift


def main():
    root = HERE.parents[1]
    out = HERE / 'private/cnf_domain_mean_alignment_01'
    if out.exists():
        raise ValueError('Preserve frozen domain alignment run')
    prepared = HERE / 'private/associated_prepared_01'
    archived = HERE / 'private/cnf_feature_challenge_01'
    full_anchor = HERE / 'private/cnf_full_anchor_challenge_01'
    anchor_path = root / 'data/E8.5_RNA.h5ad'
    target_path = root / 'data/E9.5_RNA.h5ad'
    panel_path = root / 'outputs/t1_run/T1__val.genes.txt'
    panel = panel_path.read_text().splitlines()
    seeds = [20260928, 20260929, 20260930]
    candidates = ['copy', 'unshifted', 'shift_half', 'shift_full']
    plan = {
        'created_utc': now(), 'source_sha256': {f: digest(HERE / f) for f in
            ['cnf_domain_mean_alignment.py', 'full_anchor_slope_forecast.py',
             'partial_anchor_forecast.py', 'log1p_positive_forecast.py',
             'feature_panel_forecast.py', 'offline_backtest.py', 'temporary_forecast_cache.py']},
        'input_sha256': {str(p.relative_to(root)): digest(p) for p in
                         [anchor_path, target_path, panel_path]},
        'prepared_report_sha256': digest(prepared / 'report.json'),
        'encoder_sha256': digest(archived / 'encoder4096.npz'),
        'flow_sha256': digest(archived / 'features4096.pt'),
        'baseline_report_sha256': digest(full_anchor / 'report.json'),
        'baseline_generation_sha256': digest(full_anchor / 'generation.json'),
        'cutoff': 8.5, 'target': 9.5, 'donor_seed': 20260928,
        'scoring_seeds': seeds, 'donors': 1500, 'targets_per_seed': 2000,
        'candidate_names': candidates, 'alignment_alpha': [0., .5, 1.],
        'mechanism': 'Compute frozen 8D encoder mean on sampled past source E8.5 and all permitted challenge E8.5 anchors. For each challenge donor z, integrate the original source-trained field from z-alpha*(challenge_mean-source_mean), then add that shift back. Source field, decoder and full-anchor positive/detection slopes stay fixed at .25/.75.',
        'hypothesis': 'Translation of the stage-matched source latent coordinate frame may reduce encoder domain shift without learning from future expression.',
        'controls': 'Identical E8.5 donors, target panels and scorer: persistence and exactly replayed unshifted full-anchor model.',
        'scope': 'Three reused challenge E9.5 development panels, unchanged 32285-gene scorer and calibration. No independent validation; mean alignment can erase biology and must not be interpreted causally.',
        'decision_rule': 'Consider only if one shifted candidate improves each of three panels against both controls with no mean metric-skill regression and later passes separate temporal and 64-replicate >72 gate.',
        'storage': 'D-only temporary full forecasts; retain compact plans, hashes, indices, raw metrics and genuine events.',
        'submissions_allowed': 0, 'jev_requests_allowed': 0,
    }
    out.mkdir(); (out / 'plan.json').write_text(json.dumps(plan, indent=2))
    events = out / 'events.jsonl'; append_event(events, 'plan_frozen', sha256=digest(out / 'plan.json'))
    state_path = HERE / 'LOCAL_OPTIMIZATION_STATE.json'; state = json.loads(state_path.read_text())
    state['active_jobs'] = [out.name]; state['local_process_running'] = True
    state['active_run_path'] = f'private/{out.name}'
    state['domain_mean_alignment_job'] = {'status': 'running', 'plan_sha256': digest(out / 'plan.json'),
                                          'planned_scores': 12, 'completed_scores': 0}
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
    np.save(out / 'donor_rows.npy', donor_rows)
    np.testing.assert_array_equal(donor_rows, np.load(archived / 'anchor_rows.npy'))
    with np.load(archived / 'encoder4096.npz') as encoder:
        features = encoder['features']; center = encoder['center']; scale = encoder['scale']
        net = DensityFlowNet(encoder['basis'], encoder['pca_center'], 8.5, 7.25)
    net.load_state_dict(torch.load(archived / 'features4096.pt',
                                   weights_only=False, map_location='cpu')['net'])
    net.eval()
    guard = np.load(archived / 'features.npy')
    program = FullAnchorSlopeForecast(x, stages, 8.5, donors, panel, symbols, net,
                                      center, scale, features, guard,
                                      anchor_path=anchor_path)
    program.configure(.25, .75)
    counts = Counter(symbols)
    lookup = {s: i for i, s in enumerate(symbols) if s and counts[s] == 1}
    atlas_features = np.array([lookup[panel[i]] for i in features])
    source_rows = np.flatnonzero(stages == 8.5)
    with torch.no_grad():
        source_z = net.encode(torch.tensor(
            (np.asarray(x[np.ix_(source_rows, atlas_features)])-center)/scale,
            dtype=torch.float32))[0].numpy()
    source_mean = source_z.mean(0)
    challenge_mean = program.anchor_latent_mean + program.zcenter
    delta = challenge_mean - source_mean
    np.savez_compressed(out / 'alignment.npz', source_mean=source_mean,
                        challenge_mean=challenge_mean, delta=delta)
    append_event(events, 'past_only_stage_mean_alignment_fitted',
                 source_rows=len(source_rows), challenge_rows=program.audit['anchor_calibration_rows'],
                 shift_norm=float(np.linalg.norm(delta)),
                 alignment_sha256=digest(out / 'alignment.npz'))
    old_generation = json.loads((full_anchor / 'generation.json').read_text())
    cache = TemporaryForecastCache(HERE / 'private/temporary_cache')
    generation = {}
    try:
        for name in candidates:
            if name == 'copy':
                pred, ids, audit = donors.copy(), np.arange(len(donors)), {'method': 'persistence'}
            else:
                alpha = {'unshifted': 0., 'shift_half': .5, 'shift_full': 1.}[name]
                program.net = ShiftedTrajectory(net, alpha*delta)
                pred, ids, audit = program.predict(9.5, 'joint', 1., sampling='systematic')
                audit = dict(audit, source_to_challenge_mean_alignment_strength=alpha)
            np.save(out / f'{name}_indices.npy', ids)
            generation[name] = {'prediction_sha256': cache.put(name, pred), 'audit': audit}
            if name == 'unshifted' and generation[name]['prediction_sha256'] != \
                    old_generation['full_anchor_p0.25_d0.75']['prediction_sha256']:
                raise ValueError('Unshifted forecast failed archived exact replay')
            append_event(events, 'forecast_frozen', candidate=name,
                         prediction_sha256=generation[name]['prediction_sha256'])
            del pred
        (out / 'generation.json').write_text(json.dumps(generation, indent=2))
        append_event(events, 'all_forecasts_frozen_before_target_read')
        core, _ = load_core()
        prior = json.loads((archived / 'report.json').read_text())
        report = {'status': 'running', 'plan': plan, 'generation': generation, 'panels': [],
                  'scorer_manifest_sha256': digest(HERE / 'private/scorer_source/manifest.json'),
                  'official_score': None, 'local_gate_passed': False}
        for seed in seeds:
            future, target_rows = read_cells(target_path, panel, 2000, seed)
            np.save(out / f'target_rows_{seed}.npy', target_rows)
            order = np.random.default_rng(seed).permutation(len(future))
            evaluator = Panel(core, future[order[:1000]], donors, seed)
            floor = evaluator.metrics(donors)
            ceiling = evaluator.metrics(future[order[1000:]])
            archived_panel = next(p for p in prior['panels'] if p['seed'] == seed)
            if floor != archived_panel['floor'] or ceiling != archived_panel['ceiling']:
                raise ValueError('Calibration panel changed')
            rows = []
            for name in candidates:
                with cache.read(name, consume=False) as pred:
                    raw = evaluator.metrics(pred)
                row = {'candidate': name,
                       'prediction_sha256': generation[name]['prediction_sha256'],
                       'raw_metrics': raw, **evaluator.aggregate(raw, floor, ceiling)}
                rows.append(row)
                append_event(events, 'candidate_scored', seed=seed, **row)
                state = json.loads(state_path.read_text())
                state['domain_mean_alignment_job']['completed_scores'] += 1
                state_path.write_text(json.dumps(state, indent=2))
            report['panels'].append({'seed': seed, 'cutoff': 8.5, 'target': 9.5,
                                     'floor': floor, 'ceiling': ceiling, 'results': rows,
                                     'target_rows_sha256': digest(out / f'target_rows_{seed}.npy')})
            (out / 'report.partial.json').write_text(json.dumps(report, indent=2))
        report['status'] = 'completed'
        (out / 'report.json').write_text(json.dumps(report, indent=2))
        summary = []
        for name in candidates:
            rows = [next(r for r in p['results'] if r['candidate'] == name) for p in report['panels']]
            summary.append({'candidate': name, 'scores': [r['local_score'] for r in rows],
                            'raw_metrics': [r['raw_metrics'] for r in rows],
                            'skills': [r['skills'] for r in rows],
                            'all_calibrations_valid': all(r['calibration_valid'] for r in rows)})
        controls = {x['candidate']: x for x in summary}
        passing = [x['candidate'] for x in summary if x['candidate'].startswith('shift_') and
                   x['all_calibrations_valid'] and
                   all(x['scores'][i] > max(controls['copy']['scores'][i],
                                            controls['unshifted']['scores'][i]) for i in range(3)) and
                   all(np.mean([x['skills'][i][m] for i in range(3)]) >=
                       np.mean([controls['unshifted']['skills'][i][m] for i in range(3)])
                       for m in ['de_score', 'de_direction', 'mmd_u', 'variogram'])]
        public = {'updated_utc': now(), 'status': 'completed', 'summary': summary,
                  'panels': report['panels'], 'passing_candidates': passing,
                  'source_stage_rows': len(source_rows),
                  'challenge_stage_rows': program.audit['anchor_calibration_rows'],
                  'shift_norm': float(np.linalg.norm(delta)),
                  'plan_sha256': digest(out / 'plan.json'),
                  'report_sha256': digest(out / 'report.json'),
                  'official_score': None, 'local_72_gate_passed': False,
                  'submissions_used': 0}
        (HERE / 'CNF_DOMAIN_MEAN_ALIGNMENT_RESULTS.json').write_text(json.dumps(public, indent=2))
        state = json.loads(state_path.read_text())
        state['active_jobs'] = []; state['local_process_running'] = False
        state['domain_mean_alignment_job'].update(status='completed',
            report_sha256=public['report_sha256'], passing_candidates=passing)
        state_path.write_text(json.dumps(state, indent=2))
        append_event(events, 'batch_completed', passing_candidates=passing, scores=12)
        from index_scores import main as index_scores
        index_scores()
    finally:
        cache.close()


if __name__ == '__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        main()
