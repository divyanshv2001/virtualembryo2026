"""Paired 16-resample development stability of a frozen E8.5 domain shift."""
import argparse
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
from cnf_domain_mean_alignment import ShiftedTrajectory
from full_anchor_slope_forecast import FullAnchorSlopeForecast
from temporary_forecast_cache import TemporaryForecastCache
from offline_backtest import load_core, Panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = HERE.parents[1]
    out = HERE / 'private/cnf_domain_mean_alignment_stability_01'
    previous = HERE / 'private/cnf_domain_mean_alignment_01'
    prepared = HERE / 'private/associated_prepared_01'
    archived = HERE / 'private/cnf_feature_challenge_01'
    anchor_path = root / 'data/E8.5_RNA.h5ad'
    target_path = root / 'data/E9.5_RNA.h5ad'
    panel_path = root / 'outputs/t1_run/T1__val.genes.txt'
    panel = panel_path.read_text().splitlines()
    seeds = list(range(20261001, 20261017))
    candidates = ['copy', 'unshifted', 'shift_full']
    plan = {
        'created_utc': now(), 'source_sha256': {f: digest(HERE / f) for f in
            ['cnf_domain_mean_alignment_stability.py', 'cnf_domain_mean_alignment.py',
             'full_anchor_slope_forecast.py', 'partial_anchor_forecast.py',
             'log1p_positive_forecast.py', 'feature_panel_forecast.py',
             'temporary_forecast_cache.py', 'offline_backtest.py']},
        'previous_report_sha256': digest(previous / 'report.json'),
        'previous_generation_sha256': digest(previous / 'generation.json'),
        'alignment_sha256': digest(previous / 'alignment.npz'),
        'prepared_report_sha256': digest(prepared / 'report.json'),
        'encoder_sha256': digest(archived / 'encoder4096.npz'),
        'flow_sha256': digest(archived / 'features4096.pt'),
        'input_sha256': {str(p.relative_to(root)): digest(p) for p in
                         [anchor_path, target_path, panel_path]},
        'cutoff': 8.5, 'target': 9.5, 'donor_seed': 20260928,
        'seeds': seeds, 'candidates': candidates, 'donors': 1500,
        'target_rows_per_seed': 2000, 'scored_truth_count': 1000,
        'fit': 'No new model fitting. Rebuild original past-only frozen 8D CNF and E8.5 full-anchor positive/detection .25/.75 calibration. Use the source/challenge E8.5 mean-frame shift frozen in prior run. Require bit-identical .npy forecast hashes for persistence, unshifted and full-shift versus that run.',
        'scope': 'Sixteen paired resamplings from the same exposed challenge E9.5 dataset; unchanged 32285-gene scorer, panel and metric calibration. Cell resampling is not an independent embryo or prospective temporal test.',
        'decision': 'Report all 48 scores and four metric vectors. Do not expand to a 64-replicate readiness test if shifted mean <60 or its paired 2.5th-percentile gain over unshifted <=0. Even a favorable 16-resample pilot cannot pass the >72 final gate.',
        'storage': 'D-only temporary forecasts; one prediction loaded for scoring at a time. On resume regenerate exact forecasts and skip completed seed panels.',
        'submissions_allowed': 0, 'jev_requests_allowed': 0,
    }
    if args.resume:
        prior = json.loads((out / 'plan.json').read_text())
        for key in plan:
            if key != 'created_utc' and plan[key] != prior[key]:
                raise ValueError('Resume plan mismatch: ' + key)
        plan = prior
    else:
        if out.exists():
            raise ValueError('Preserve prior stability run; use --resume')
        out.mkdir(); (out / 'plan.json').write_text(json.dumps(plan, indent=2))
    events = out / 'events.jsonl'
    append_event(events, 'run_resumed' if args.resume else 'plan_frozen',
                 sha256=digest(out / 'plan.json'))
    partial = out / 'report.partial.json'
    if args.resume and partial.exists():
        report = json.loads(partial.read_text())
    else:
        report = {'status': 'running', 'plan': plan, 'panels': [], 'generation': {},
                  'scorer_manifest_sha256': digest(HERE / 'private/scorer_source/manifest.json'),
                  'official_score': None, 'local_gate_passed': False}
    state_path = HERE / 'LOCAL_OPTIMIZATION_STATE.json'; state = json.loads(state_path.read_text())
    state['active_jobs'] = [out.name]; state['local_process_running'] = True
    state['active_run_path'] = f'private/{out.name}'
    state['domain_alignment_stability_job'] = {
        'status': 'running', 'planned_scores': 48,
        'completed_scores': len(report['panels'])*3,
        'plan_sha256': digest(out / 'plan.json')}
    state_path.write_text(json.dumps(state, indent=2))
    prepared_report = json.loads((prepared / 'report.json').read_text())
    for name, key in [('expression.npy', 'expression_sha256'),
                      ('selected_metadata.csv', 'metadata_sha256'), ('genes.csv', 'genes_sha256')]:
        if digest(prepared / name) != prepared_report[key]:
            raise ValueError('Prepared input changed: ' + name)
    x = np.load(prepared / 'expression.npy', mmap_mode='r')
    stages = pd.read_csv(prepared / 'selected_metadata.csv').numeric_stage.to_numpy(float)
    symbols = pd.read_csv(prepared / 'genes.csv').symbol.fillna('').tolist()
    donors, donor_rows = read_cells(anchor_path, panel, 1500, plan['donor_seed'])
    if args.resume:
        np.testing.assert_array_equal(donor_rows, np.load(out / 'donor_rows.npy'))
    else:
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
    with np.load(previous / 'alignment.npz') as saved:
        delta = saved['delta']
    expected = json.loads((previous / 'generation.json').read_text())
    cache = TemporaryForecastCache(HERE / 'private/temporary_cache')
    try:
        generation = {}
        for name in candidates:
            if name == 'copy':
                pred, ids, audit = donors.copy(), np.arange(len(donors)), {'method': 'persistence'}
            else:
                alpha = 0. if name == 'unshifted' else 1.
                program.net = ShiftedTrajectory(net, alpha*delta)
                pred, ids, audit = program.predict(9.5, 'joint', 1., sampling='systematic')
                audit = dict(audit, source_to_challenge_mean_alignment_strength=alpha)
            indices_path = out / f'{name}_indices.npy'
            if args.resume and indices_path.exists():
                np.testing.assert_array_equal(ids, np.load(indices_path))
            else:
                np.save(indices_path, ids)
            sha = cache.put(name, pred)
            if sha != expected[name]['prediction_sha256']:
                raise ValueError('Frozen forecast hash mismatch: ' + name)
            generation[name] = {'prediction_sha256': sha, 'audit': audit}
            append_event(events, 'forecast_replayed_exactly', candidate=name,
                         prediction_sha256=sha)
            del pred
        report['generation'] = generation
        partial.write_text(json.dumps(report, indent=2))
        append_event(events, 'all_forecasts_frozen_before_target_read')
        core, _ = load_core()
        for seed in seeds:
            if any(p['seed'] == seed for p in report['panels']):
                continue
            future, rows = read_cells(target_path, panel, 2000, seed)
            np.save(out / f'target_rows_{seed}.npy', rows)
            order = np.random.default_rng(seed).permutation(len(future))
            evaluator = Panel(core, future[order[:1000]], donors, seed)
            floor = evaluator.metrics(donors)
            ceiling = evaluator.metrics(future[order[1000:]])
            results = []
            for name in candidates:
                with cache.read(name, consume=False) as pred:
                    raw = evaluator.metrics(pred)
                row = {'candidate': name, 'prediction_sha256': generation[name]['prediction_sha256'],
                       'raw_metrics': raw, **evaluator.aggregate(raw, floor, ceiling)}
                results.append(row)
                append_event(events, 'candidate_scored', seed=seed, **row)
            report['panels'].append({'seed': seed, 'cutoff': 8.5, 'target': 9.5,
                                     'floor': floor, 'ceiling': ceiling, 'results': results,
                                     'target_rows_sha256': digest(out / f'target_rows_{seed}.npy')})
            partial.write_text(json.dumps(report, indent=2))
            state = json.loads(state_path.read_text())
            state['domain_alignment_stability_job']['completed_scores'] = len(report['panels'])*3
            state_path.write_text(json.dumps(state, indent=2))
            del future, evaluator
        report['status'] = 'completed'
        (out / 'report.json').write_text(json.dumps(report, indent=2))
        summaries = []
        for name in candidates:
            outcomes = [next(r for r in p['results'] if r['candidate'] == name)
                        for p in report['panels']]
            score = np.asarray([r['local_score'] for r in outcomes])
            summaries.append({'candidate': name, 'scores': score.tolist(),
                              'mean_score': float(score.mean()),
                              'lower_tail_score': float(np.quantile(score, .025)),
                              'all_calibrations_valid': all(r['calibration_valid'] for r in outcomes),
                              'mean_skills': {m: float(np.mean([r['skills'][m] for r in outcomes]))
                                              for m in outcomes[0]['skills']}})
        by_name = {x['candidate']: x for x in summaries}
        paired = np.asarray(by_name['shift_full']['scores']) - np.asarray(by_name['unshifted']['scores'])
        expand = bool(by_name['shift_full']['mean_score'] >= 60 and
                      np.quantile(paired, .025) > 0 and
                      by_name['shift_full']['all_calibrations_valid'])
        public = {'updated_utc': now(), 'status': 'completed',
                  'summaries': summaries, 'paired_gain_vs_unshifted': {
                      'mean': float(paired.mean()),
                      'lower_tail_2p5': float(np.quantile(paired, .025)),
                      'positive_replicates': int((paired > 0).sum()),
                      'replicates': len(paired)},
                  'expand_to_final_64_replicates': expand,
                  'panels': report['panels'], 'plan_sha256': digest(out / 'plan.json'),
                  'report_sha256': digest(out / 'report.json'),
                  'official_score': None, 'local_72_gate_passed': False,
                  'submissions_used': 0}
        (HERE / 'CNF_DOMAIN_ALIGNMENT_STABILITY_RESULTS.json').write_text(json.dumps(public, indent=2))
        state = json.loads(state_path.read_text())
        state['active_jobs'] = []; state['local_process_running'] = False
        state['domain_alignment_stability_job'].update(
            status='completed', report_sha256=public['report_sha256'],
            expand_to_final_64_replicates=expand)
        state_path.write_text(json.dumps(state, indent=2))
        append_event(events, 'stability_batch_completed', scores=48,
                     expand_to_final_64_replicates=expand)
        from index_scores import main as index_scores
        index_scores()
    finally:
        cache.close()


if __name__ == '__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        main()
