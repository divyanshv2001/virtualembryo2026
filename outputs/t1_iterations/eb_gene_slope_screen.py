"""Past-only source-stage screen for a soft empirical-Bayes gene-slope formula.

Gene-rank proxies here are not the full-panel T1 score. The screen rejects
weak priors before making any challenge-domain forecast.
"""
import json
from collections import Counter

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import rankdata

from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event


STAGES = [7.5, 7.75, 8.0, 8.25, 8.5]
FOLDS = [(8.0, 8.25), (8.25, 8.5)]
K = 200
ALPHAS = [.25, .5]
GAMMAS = [0, 1]


def signed_overlap(delta, up, down):
    order = np.argsort(-delta, kind='stable')
    found = len(np.intersect1d(order[:len(up)], up)) + len(np.intersect1d(order[-len(down):], down))
    return found / (len(up) + len(down))


def partial_rank_direction(pred, truth, ref):
    ranks = np.vstack([rankdata(a) for a in (pred, truth, ref)])
    corr = np.corrcoef(ranks)
    if not np.isfinite(corr).all() or 1 - abs(corr[0, 2]) < 1e-6:
        return 0.
    numerator = corr[0, 1] - corr[0, 2] * corr[1, 2]
    denominator = np.sqrt(max((1-corr[0, 2]**2)*(1-corr[1, 2]**2), 1e-12))
    return float(np.clip(numerator/denominator, -1., 1.))


def score(pred, truth, ref, up, down):
    chance = max(signed_overlap(ref, up, down), signed_overlap(-ref, up, down))
    overlap = signed_overlap(pred, up, down)
    return {'signed_top200_overlap': float(overlap),
            'reference_magnitude_chance': float(chance),
            'chance_adjusted_overlap': float((overlap-chance)/(1-chance)),
            'partial_spearman': partial_rank_direction(pred, truth, ref)}


def main():
    out = HERE / 'private/eb_gene_slope_screen_01'
    if out.exists():
        raise ValueError('Preserve frozen EB screen; do not overwrite')
    source = HERE / 'private/associated_prepared_01'
    report = json.loads((source / 'report.json').read_text())
    for name, key in [('expression.npy', 'expression_sha256'),
                      ('selected_metadata.csv', 'metadata_sha256'), ('genes.csv', 'genes_sha256')]:
        if digest(source / name) != report[key]:
            raise ValueError('Prepared source changed: ' + name)
    panel_path = HERE.parents[1] / 'outputs/t1_run/T1__val.genes.txt'
    panel = panel_path.read_text().splitlines()
    symbols = pd.read_csv(source / 'genes.csv').symbol.fillna('').tolist()
    counts = Counter(symbols)
    lookup = {s: i for i, s in enumerate(symbols) if s and counts[s] == 1}
    mapped = np.array([i for i, s in enumerate(panel) if s in lookup], dtype=int)
    atlas = np.array([lookup[panel[i]] for i in mapped], dtype=int)
    plan = {'created_utc': now(), 'source_sha256': digest(HERE / 'eb_gene_slope_screen.py'),
            'prepared_report_sha256': digest(source / 'report.json'),
            'panel_sha256': digest(panel_path), 'stages_read': STAGES,
            'folds': FOLDS, 'mapped_genes': len(mapped), 'truth_top_up_down': K,
            'alphas': ALPHAS, 'gammas': GAMMAS,
            'estimator': 'Recent .25-day per-gene pseudobulk slope b; s2 = cell-sampling variance of b + .25*(b-previous_slope)^2. Empirical zero-centered normal prior variance tau2=max(var(b)-median(s2),0). Posterior m=tau2/(tau2+s2)*b, posterior sd=sqrt(tau2*s2/(tau2+s2)), certainty q=abs(2Phi(m/sd)-1). Forecast delta h*((1-alpha)*b+alpha*m*q^gamma), h=.25. Baseline alpha0 is recent linear slope.',
            'proxy_metric': 'Fixed top200 truth up/down signed-rank chance-adjusted overlap and partial Spearman controlling reference mean, mapped genes only. These are screens, not full T1 scorer or official skill.',
            'selection_rule': 'Advance an EB setting to expensive full-panel work only if it strictly improves chance-adjusted overlap on both historical source folds with no partial-Spearman regression on either. Never select using challenge E9.5 target.',
            'scope': 'Two historical source folds with training <= cutoff. No challenge E9.5 reads, full-panel score, future-stage fitting, Jev call or official submission.'}
    out.mkdir()
    (out / 'plan.json').write_text(json.dumps(plan, indent=2))
    events = out / 'events.jsonl'
    append_event(events, 'plan_frozen', sha256=digest(out / 'plan.json'))
    x = np.load(source / 'expression.npy', mmap_mode='r')
    stages = pd.read_csv(source / 'selected_metadata.csv').numeric_stage.to_numpy(float)
    stats = {}
    for stage in STAGES:
        rows = np.flatnonzero(stages == stage)
        if len(rows) < 100:
            raise ValueError(f'Insufficient source stage {stage}')
        mean = np.empty(len(mapped)); var = np.empty(len(mapped))
        for start in range(0, len(mapped), 256):
            sl = slice(start, min(start+256, len(mapped)))
            values = np.asarray(x[np.ix_(rows, atlas[sl])], dtype=np.float32)
            mean[sl] = values.mean(0, dtype=np.float64)
            var[sl] = values.var(0, dtype=np.float64, ddof=1)
        stats[stage] = {'n': len(rows), 'mean': mean, 'var': var}
        append_event(events, 'stage_sufficient_statistics', stage=stage, rows=len(rows))
    outcomes = []
    prediction_arrays = {}
    for cutoff, target in FOLDS:
        prev = cutoff-.25; earlier = cutoff-.5; h = target-cutoff
        ref = stats[cutoff]['mean']; observed = stats[target]['mean']-ref
        b = (ref-stats[prev]['mean'])/.25
        b_prev = (stats[prev]['mean']-stats[earlier]['mean'])/.25
        sampling_s2 = (stats[cutoff]['var']/stats[cutoff]['n']+
                       stats[prev]['var']/stats[prev]['n'])/.25**2
        s2 = sampling_s2 + .25*(b-b_prev)**2
        tau2 = max(float(np.var(b)-np.median(s2)), 1e-8)
        weight = tau2/(tau2+s2)
        posterior_mean = weight*b
        posterior_sd = np.sqrt(np.maximum(weight*s2, 1e-12))
        certainty = np.abs(2*ndtr(posterior_mean/posterior_sd)-1)
        truth_order = np.argsort(observed, kind='stable')
        up, down = truth_order[-K:], truth_order[:K]
        candidates = {'recent_linear': h*b}
        for alpha in ALPHAS:
            for gamma in GAMMAS:
                candidates[f'eb_a{alpha}_g{gamma}'] = h*((1-alpha)*b+alpha*posterior_mean*certainty**gamma)
        results = {name: score(pred, observed, ref, up, down)
                   for name, pred in candidates.items()}
        for name, pred in candidates.items():
            prediction_arrays[f'{cutoff}_{name}'] = pred.astype(np.float32)
        outcomes.append({'cutoff': cutoff, 'target': target,
                         'source_stage_rows': {str(t): stats[t]['n'] for t in (earlier,prev,cutoff,target)},
                         'tau2': tau2, 'median_s2': float(np.median(s2)),
                         'median_posterior_weight': float(np.median(weight)),
                         'median_sign_certainty': float(np.median(certainty)),
                         'results': results})
        append_event(events, 'historical_fold_scored', cutoff=cutoff, target=target,
                     tau2=tau2, results=results)
    np.savez_compressed(out / 'frozen_gene_deltas.npz', mapped=mapped, **prediction_arrays)
    names = [name for name in outcomes[0]['results'] if name != 'recent_linear']
    passing = [name for name in names if all(
        fold['results'][name]['chance_adjusted_overlap'] > fold['results']['recent_linear']['chance_adjusted_overlap']
        and fold['results'][name]['partial_spearman'] >= fold['results']['recent_linear']['partial_spearman']
        for fold in outcomes)]
    full = {'status': 'completed', 'plan': plan, 'outcomes': outcomes,
            'passing_candidates': passing, 'prediction_deltas_sha256': digest(out / 'frozen_gene_deltas.npz'),
            'local_full_panel_scores': 0, 'official_submissions': 0}
    (out / 'report.json').write_text(json.dumps(full, indent=2))
    public = {'updated_utc': now(), 'status': 'completed', 'outcomes': outcomes,
              'passing_candidates': passing, 'plan_sha256': digest(out / 'plan.json'),
              'report_sha256': digest(out / 'report.json'),
              'scope': plan['scope'], 'not_full_panel_score': True}
    (HERE / 'EB_GENE_SLOPE_SCREEN_RESULTS.json').write_text(json.dumps(public, indent=2))
    append_event(events, 'screen_completed', passing_candidates=passing,
                 report_sha256=public['report_sha256'])
    print(json.dumps({'passing_candidates': passing, 'outcomes': outcomes,
                      'report_sha256': public['report_sha256']}))


if __name__ == '__main__':
    main()
