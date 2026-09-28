"""Training-only expression-state matching and regularized multiplicative drift.

No provided embeddings, target data, external atlases, API calls or uploads.
Cell splits diagnose estimator stability, not future-stage accuracy.
"""
import argparse
import json
import sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import eigh
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
T1 = ROOT / 'outputs/t1_run'
sys.path.insert(0, str(T1))
from run_t1 import digest, validate
from iterate import append_event, now


def select_features(matrices, count):
    """Pooled log-expression variance; no labels or external feature lists."""
    n = sum(x.shape[0] for x in matrices)
    mean = sum(np.asarray(x.sum(axis=0)).ravel() for x in matrices) / n
    second = sum(np.asarray(x.power(2).sum(axis=0)).ravel() for x in matrices) / n
    variance = np.maximum(second - mean * mean, 0)
    selected = np.argsort(-variance, kind='stable')[:count]
    return np.sort(selected), variance


def latent_space(matrices, features, dimensions):
    blocks = [x[:, features].toarray().astype(np.float64) for x in matrices]
    joined = np.vstack(blocks)
    center = joined.mean(axis=0)
    scale = np.maximum(joined.std(axis=0), .1)
    joined = np.clip((joined - center) / scale, -10, 10)
    # Center after clipping so that PCA is centered on the fitted representation.
    clipped_center = joined.mean(axis=0)
    joined -= clipped_center
    cov = joined.T @ joined / max(len(joined) - 1, 1)
    values, vectors = eigh(cov, subset_by_index=[len(features)-dimensions, len(features)-1])
    vectors = vectors[:, ::-1]
    z = (joined @ vectors).astype(np.float32)
    split = matrices[0].shape[0]
    return [z[:split], z[split:]], {'center': center, 'scale': scale,
        'clipped_center': clipped_center, 'components': vectors,
        'eigenvalues': values[::-1]}


def regularized_log_ratio(mean_a, mean_b, se2, strength=.5, cap=2.):
    """Shrink noisy local changes, then bound each gene's abundance multiplier."""
    delta = mean_b - mean_a
    reliability = delta**2 / (delta**2 + se2 + 1e-12)
    # X is log-normalized. These are local mean expm1(X) values, not raw counts.
    ratio = np.log((mean_b + .05) / (mean_a + .05))
    return np.clip(strength * reliability * ratio, -np.log(cap), np.log(cap))


def apply_ratio(x, log_ratio):
    y = np.log1p(np.expm1(np.asarray(x, dtype=np.float64)) * np.exp(log_ratio))
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Invalid transformed expression')
    return y.astype(np.float32)


def neighborhood_stats(x, rows):
    block = x[rows].copy()
    block.data = np.expm1(block.data)
    mean = np.asarray(block.sum(axis=0)).ravel() / len(rows)
    second = np.asarray(block.power(2).sum(axis=0)).ravel() / len(rows)
    se2 = np.maximum(second - mean**2, 0) / (len(rows)-1)
    return mean, se2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--round', default='round_03_transport')
    p.add_argument('--features', type=int, default=768)
    p.add_argument('--dimensions', type=int, default=24)
    p.add_argument('--neighbors', type=int, default=64)
    args = p.parse_args()
    if not args.round.replace('_', '').isalnum(): raise ValueError('Invalid round')
    if not 2 <= args.dimensions < args.features <= 2048: raise ValueError('Invalid feature dimensions')
    if not 8 <= args.neighbors <= 256: raise ValueError('Invalid neighborhood size')
    out = HERE / 'private' / args.round
    if out.exists(): raise SystemExit('Round exists; use a new name.')
    out.mkdir(parents=True)
    events = out / 'batch_events.jsonl'
    source_report = json.loads((T1 / 'run_report.json').read_text())
    donors = np.load(T1 / 'donor_rows.npy', allow_pickle=False)
    config = {'method': 'local_state_transport', 'features': args.features,
        'dimensions': args.dimensions, 'neighbors': args.neighbors,
        'strength': .5, 'multiplier_cap': 2., 'seed': 20260928,
        'support': 'Donor zeros remain zero; no new expression support generated.'}
    plan = {'created_before_execution_utc': now(), 'config': config,
        'input_files': source_report['source_files'], 'code_sha256': digest(Path(__file__)),
        'donor_sha256': digest(T1 / 'donor_rows.npy'), 'agent_team_configuration_lock': False,
        'scope': 'Training-cell stability diagnostics are not E10.5 forecast validation.',
        'live_jev_requests': 0}
    (out / 'batch_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    append_event(events, 'plan_frozen', sha256=digest(out / 'batch_plan.json'))
    panel = (T1 / 'T1__val.genes.txt').read_text().splitlines()
    spec = json.loads((T1 / 'index.json').read_text())['T1:val']
    matrices = []
    for source in source_report['source_files']:
        path = ROOT / source['path']
        if digest(path) != source['sha256']: raise ValueError('Source changed')
        a = ad.read_h5ad(path)
        columns = a.var_names.get_indexer(panel)
        if (columns < 0).any(): raise ValueError('Missing panel genes')
        matrices.append(sparse.csr_matrix(a.X[:, columns], dtype=np.float64))
        del a
    append_event(events, 'training_loaded', shapes=[list(x.shape) for x in matrices])
    features, variance = select_features(matrices, args.features)
    z, model = latent_space(matrices, features, args.dimensions)
    np.savez_compressed(out / 'representation.npz', features=features, variance=variance, **model)
    append_event(events, 'representation_fitted', features=args.features, dimensions=args.dimensions)
    rng = np.random.default_rng(config['seed'])
    # Independent reference halves. Donors never enter either late-stage neighbor pool.
    late_pool = np.setdiff1d(np.arange(len(z[1])), donors)
    pools = []
    for pool in [np.arange(len(z[0])), late_pool]:
        shuffled = rng.permutation(pool)
        pools.append(np.array_split(shuffled, 2))
    query = z[1][donors]
    neighbor_sets = []
    distances = []
    for stage in range(2):
        sets = []; ds = []
        for pool in pools[stage]:
            d, index = cKDTree(z[stage][pool]).query(query, k=args.neighbors, workers=2)
            sets.append(pool[index]); ds.append(d.mean(axis=1))
        neighbor_sets.append(sets); distances.append(np.mean(ds, axis=0))
    # Discount early-stage matches substantially farther away than same-stage matches.
    trust = np.minimum(1., (distances[1] / np.maximum(distances[0], 1e-8))**2)
    append_event(events, 'neighbors_matched', median_match_trust=float(np.median(trust)))
    # Materialize output one row at a time; retain bounded sparse training matrices.
    original = matrices[1][donors].toarray().astype(np.float32)
    prediction = np.empty_like(original)
    agreement = []; effective = []; changes = []
    for i in range(len(donors)):
        half_ratios = []
        for half in range(2):
            ma, va = neighborhood_stats(matrices[0], neighbor_sets[0][half][i])
            mb, vb = neighborhood_stats(matrices[1], neighbor_sets[1][half][i])
            half_ratios.append(regularized_log_ratio(ma, mb, va+vb,
                config['strength'], config['multiplier_cap']))
        consistent = half_ratios[0] * half_ratios[1] > 0
        ratio = np.where(consistent, (half_ratios[0]+half_ratios[1])*.5, 0) * trust[i]
        prediction[i] = apply_ratio(original[i], ratio)
        active = original[i] > 0
        agreement.append(float(consistent[active].mean()))
        effective.append(float(np.mean(np.abs(ratio[active]))))
        changes.append(float(np.mean(np.abs(prediction[i]-original[i]))))
        if (i+1) % 250 == 0: append_event(events, 'cells_transformed', cells=i+1)
    artifact = out / 'T1_val__local_state_transport.h5ad'
    ad.AnnData(X=sparse.csr_matrix(prediction),
        obs=pd.DataFrame(index=[f'prediction_{i:05}' for i in range(len(donors))]),
        var=pd.DataFrame(index=panel)).write_h5ad(artifact, compression='gzip')
    check = validate(artifact, panel, spec)
    if not np.array_equal(prediction == 0, original == 0): raise ValueError('Zero support changed')
    report = {'plan': plan, 'artifact': artifact.relative_to(ROOT).as_posix(),
        'validation': check, 'diagnostics': {
            'median_match_trust': float(np.median(trust)),
            'median_active_gene_split_sign_agreement': float(np.median(agreement)),
            'mean_active_gene_absolute_log_multiplier': float(np.mean(effective)),
            'mean_absolute_expression_change': float(np.mean(changes)),
            'donor_zero_support_preserved': True,
            'scope': 'Estimator stability and output checks only; no held-out-stage accuracy.'},
        'limitations': ['No independent embryos or E10.5 truth available.',
            'Nearest expression states do not establish lineage ancestry.',
            'Latent alignment can mix related populations or dissection effects.',
            'Preserved zeros prohibit activation of previously unexpressed genes.',
            'Only two measured stages; regularization strength is a hypothesis.'],
        'uploaded': False, 'official_score': None, 'live_jev_requests': 0}
    for source in source_report['source_files']:
        if digest(ROOT / source['path']) != source['sha256']: raise ValueError('Source modified')
    (out / 'iteration_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (out / 'score_ledger.json').write_text(json.dumps({'observations': []}), encoding='utf-8')
    append_event(events, 'candidate_validated', sha256=check['sha256'], report_sha256=digest(out / 'iteration_report.json'))


if __name__ == '__main__': main()
