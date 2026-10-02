"""Unique exact replay diagnostic; no new fitting choice, scoring, or future target use."""
import hashlib
import json
import traceback
from pathlib import Path
from collections import Counter

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from threadpoolctl import threadpool_limits

from cnf_manifold_flow import DensityFlowNet
from cnf_domain_mean_alignment import ShiftedTrajectory
from cnf_covariance_alignment import CovarianceAlignedTrajectory
from full_anchor_slope_forecast import FullAnchorSlopeForecast
from hurdle_backtest import read_cells
from temporary_forecast_cache import TemporaryForecastCache
from run_t1 import digest
from iterate import now, append_event

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = HERE / 'private/one_day_decoder_diagnostic_01'
PUBLIC = HERE / 'ONE_DAY_DECODER_DIAGNOSTIC_RESULTS.json'
NAMES = ['copy', 'mean_shift', 'covariance_0.25']


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def main():
    if RUN.exists() or PUBLIC.exists():
        raise ValueError('Preserve existing diagnostic; never duplicate')
    RUN.mkdir(parents=True)
    events = RUN / 'events.jsonl'
    cache = TemporaryForecastCache(HERE / 'private/temporary_cache')
    try:
        spec = HERE / 'NEXT_ONE_DAY_DECODER_DIAGNOSTIC.json'
        old = HERE / 'private/cnf_covariance_alignment_01'
        archived = HERE / 'private/cnf_feature_challenge_01'
        previous = HERE / 'private/cnf_domain_mean_alignment_01'
        prepared = HERE / 'private/associated_prepared_01'
        plan = json.loads((old / 'plan.json').read_text())
        frozen = json.loads((old / 'generation.json').read_text())
        for name, expected in plan['source_sha256'].items():
            if digest(HERE / name) != expected:
                raise ValueError('Frozen source changed: ' + name)
        for path, expected in [(archived / 'encoder4096.npz', plan['encoder_sha256']),
                               (archived / 'features4096.pt', plan['flow_sha256'])]:
            if digest(path) != expected:
                raise ValueError('Frozen checkpoint changed')
        for name, expected in plan['input_sha256'].items():
            if digest(ROOT / name) != expected:
                raise ValueError('Frozen input changed: ' + name)
        save(RUN / 'plan.json', {'spec_sha256': digest(spec), 'historical_plan_sha256': digest(old / 'plan.json'),
                               'source_sha256': digest(Path(__file__)), 'created_utc': now(),
                               'cutoff': 8.5, 'target': 9.5, 'controls': NAMES, 'new_scores': 0})
        append_event(events, 'plan_frozen', sha256=digest(RUN / 'plan.json'))
        panel = (ROOT / 'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
        donors, rows = read_cells(ROOT / 'data/E8.5_RNA.h5ad', panel, 1500, 20260928)
        np.testing.assert_array_equal(rows, np.load(old / 'donor_rows.npy'))
        prep = json.loads((prepared / 'report.json').read_text())
        for name, key in [('expression.npy','expression_sha256'),('genes.csv','genes_sha256'),('selected_metadata.csv','metadata_sha256')]:
            if digest(prepared / name) != prep[key]:
                raise ValueError('Prepared source changed: ' + name)
        x = np.load(prepared / 'expression.npy', mmap_mode='r')
        stages = pd.read_csv(prepared / 'selected_metadata.csv').numeric_stage.to_numpy(float)
        symbols = pd.read_csv(prepared / 'genes.csv').symbol.fillna('').tolist()
        with np.load(archived / 'encoder4096.npz') as e:
            features, center, scale = e['features'], e['center'], e['scale']
            net = DensityFlowNet(e['basis'], e['pca_center'], 8.5, 7.25)
        net.load_state_dict(torch.load(archived / 'features4096.pt', weights_only=False, map_location='cpu')['net'])
        net.eval()
        program = FullAnchorSlopeForecast(x, stages, 8.5, donors, panel, symbols, net, center, scale,
                                         features, np.load(archived / 'features.npy'), anchor_path=ROOT / 'data/E8.5_RNA.h5ad')
        program.configure(.25, .75)
        with np.load(previous / 'alignment.npz') as a:
            delta, source_mean, challenge_mean = a['delta'], a['source_mean'], a['challenge_mean']
        with np.load(old / 'alignment.npz') as a:
            transform = np.eye(len(delta)) + .25 * (a['full_map'] - np.eye(len(delta)))
        generation = {}
        for name in NAMES:
            if name == 'copy':
                pred, ids = donors.copy(), np.arange(len(donors))
            else:
                program.net = ShiftedTrajectory(net, delta) if name == 'mean_shift' else CovarianceAlignedTrajectory(net, source_mean, challenge_mean, transform)
                pred, ids, _ = program.predict(9.5, 'joint', 1., sampling='systematic')
            np.testing.assert_array_equal(ids, np.load(old / (name + '_indices.npy')))
            sha = cache.put(name, pred)
            if sha != frozen[name]['prediction_sha256']:
                raise ValueError('Exact original prediction replay failed: ' + name)
            generation[name] = {'prediction_sha256': sha, 'exact_replay': True}
            append_event(events, 'forecast_replayed', candidate=name, prediction_sha256=sha)
            del pred
        append_event(events, 'all_forecasts_verified_before_target_expression')
        counts = Counter(symbols)
        mapped = np.array([bool(g and counts[g] == 1) for g in panel])
        reference_mean = donors.mean(0, dtype=np.float64)
        reference_detection = (donors > 0).mean(0)
        historical = json.loads((old / 'report.json').read_text())
        diagnostics = []
        for seed in plan['scoring_seeds']:
            future, target_rows = read_cells(ROOT / 'data/E9.5_RNA.h5ad', panel, 2000, seed)
            np.testing.assert_array_equal(target_rows, np.load(old / f'target_rows_{seed}.npy'))
            order = np.random.default_rng(seed).permutation(len(future))
            truth = future[order[:1000]]
            mean = truth.mean(0, dtype=np.float64)
            detection = (truth > 0).mean(0)
            variance = truth.var(0, dtype=np.float64)
            genuine = mean - reference_mean
            for name in NAMES:
                with cache.read(name, consume=False) as pred:
                    pm = pred.mean(0, dtype=np.float64)
                    pred_detection = (pred > 0).mean(0)
                    pv = pred.var(0, dtype=np.float64)
                    predicted = pm - reference_mean
                    diagnostics.append({'seed': seed, 'candidate': name, 'strata': {
                        label: {'genes': int(mask.sum()), 'mean_absolute_expression_error': float(np.mean(np.abs(pm[mask] - mean[mask]))),
                                'mean_absolute_detection_error': float(np.mean(np.abs(pred_detection[mask] - detection[mask]))),
                                'mean_absolute_variance_error': float(np.mean(np.abs(pv[mask] - variance[mask]))),
                                'predicted_detection_change_sum': float((pred_detection[mask] - reference_detection[mask]).sum()),
                                'actual_detection_change_sum': float((detection[mask] - reference_detection[mask]).sum()),
                                'gene_mean_direction_agreement_epsilon1e6': float(np.mean(np.sign(np.where(abs(predicted[mask]) > 1e-6, predicted[mask], 0)) == np.sign(np.where(abs(genuine[mask]) > 1e-6, genuine[mask], 0))))}
                        for label, mask in [('all',np.ones(len(panel),dtype=bool)),('mapped',mapped),('protected',~mapped)]}})
                    np.savez_compressed(RUN / f'gene_diagnostic_{seed}_{name}.npz', target_mean=mean, target_detection=detection,
                                        predicted_mean=pm, predicted_detection=pred_detection, reference_mean=reference_mean, reference_detection=reference_detection)
            append_event(events, 'historical_panel_diagnosed', seed=seed)
        report = {'status': 'completed', 'completed_utc': now(), 'plan_sha256': digest(RUN / 'plan.json'),
                  'generation': generation, 'diagnostics': diagnostics,
                  'historical_score_report_sha256': digest(old / 'report.json'),
                  'folds': [dict(p, results=[r for r in p['results'] if r['candidate'] in NAMES]) for p in historical['panels']],
                  'scores_are_historical_replays': True, 'new_scores': 0, 'reward': 0,
                  'scope': 'Exact original forecast reconstruction; diagnostic gene-space means/detection/variance are not new DE/MMD/variogram scores. Historical panels reused and overlapping. Original raw4/skills4/calibration copied as evidence only. No new fitting choice or E10.5 truth.',
                  'passing_candidates': None}
    except Exception as exc:
        (RUN / 'traceback.txt').write_text(traceback.format_exc())
        report = {'status':'failed','error':type(exc).__name__ + ': ' + str(exc), 'new_scores':0,'reward':0,
                  'scope':'Unique diagnostic failed; preserve all evidence. No readiness claim.'}
    finally:
        cache.close()
    save(RUN / 'report.json', report)
    report['report_sha256'] = digest(RUN / 'report.json')
    save(PUBLIC, report)
    append_event(events, 'diagnostic_finished', status=report['status'], report_sha256=report['report_sha256'])
    print(json.dumps({'status':report['status'],'report':str(PUBLIC)}))


if __name__ == '__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        main()
