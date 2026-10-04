"""Observed E8.5 support of frozen residual fields; no fitting or scoring."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from train_extended_atlas import HERE
from run_t1 import digest
from hurdle_backtest import read_cells
from cnf_manifold_flow import DensityFlowNet
from graph_kinetic_residual import KineticFlow
from scalar_potential_residual import PotentialResidualFlow


def main():
    run = HERE / 'private/potential_transfer_support_audit_01'
    report = HERE / 'POTENTIAL_TRANSFER_SUPPORT_AUDIT_RESULTS.json'
    if run.exists() or report.exists():
        raise ValueError('Never duplicate audit')
    frozen = HERE / 'private/scalar_potential_residual_past_fold_01'
    old_plan = json.loads((frozen / 'plan.json').read_text())
    for name in ['scalar_potential_residual.py', 'cnf_manifold_flow.py', 'cnf_density_flow.py', 'graph_kinetic_residual.py']:
        if digest(HERE / name) != old_plan['source_sha256'][name]:
            raise ValueError('Frozen mechanism changed')
    root = HERE.parents[1]
    anchor = root / 'data/E8.5_RNA.h5ad'
    panel_path = root / 'outputs/t1_run/T1__val.genes.txt'
    data = HERE / 'private/associated_prepared_01'
    for name, expected in old_plan['input_sha256'].items():
        if digest(data / name) != expected:
            raise ValueError('Prepared source changed')
    bp = HERE / 'private/graph_kinetics_past_fold_repair_01/kinetic_none400.pt'
    paths = [Path(__file__), anchor, panel_path, frozen / 'fresh_encoder.npz', bp] + [frozen / ('residual_' + k + '400.pt') for k in ['vector', 'potential']]
    pins = {str(p): digest(p) for p in paths}
    run.mkdir(parents=True)
    plan = {'scope': 'New frozen-potential field support diagnostic at observed E8.5; existing gross-scale/lineage/mean-alignment audits not repeated. No E9.5 expression, model fitting, full-panel forecasts, score or reward.', 'sha256': pins, 'seed': 20260928, 'sample_per_domain': 1500, 'fit_cutoff': 8.25, 'observed_stage': 8.5}
    (run / 'plan.json').write_text(json.dumps(plan, indent=2))
    panel = panel_path.read_text().splitlines()
    donors, _ = read_cells(anchor, panel, 1500, 20260928)
    with np.load(frozen / 'fresh_encoder.npz') as e:
        encoder = {k: e[k] for k in e.files}
    symbols = pd.read_csv(data / 'genes.csv').symbol.fillna('').tolist()
    symbol_rows = {s: i for i, s in enumerate(symbols) if s and symbols.count(s) == 1}
    columns = np.array([symbol_rows[panel[i]] for i in encoder['features']])
    stages = pd.read_csv(data / 'selected_metadata.csv').numeric_stage.to_numpy(float)
    rows = np.flatnonzero(stages == 8.5)
    rows = np.random.default_rng(20260928).choice(rows, 1500, replace=False)
    x = np.load(data / 'expression.npy', mmap_mode='r')
    source = np.asarray(x[np.ix_(rows, columns)], np.float32)
    state = torch.load(bp, weights_only=False, map_location='cpu')['net']
    drift = DensityFlowNet(encoder['basis'], encoder['pca_center'], 8.25, 7.25)
    base = KineticFlow(drift, state['gene_mean'].numpy(), state['edges'][1].numpy(), state['edges'][0].numpy(), 'none')
    base.load_state_dict(state)
    with np.load(frozen / 'residual_past_context.npz') as c:
        tree = cKDTree(c['coordinates'])
    results = {}
    for domain, values in [('source', source), ('challenge', donors[:, encoder['features']])]:
        with torch.no_grad():
            z = base.encode(torch.tensor((values - encoder['center']) / encoder['scale']))[0]
            velocity = base.velocity(1.25, z)
        nearest = tree.query(z.numpy(), k=1)[0]
        fields = {}
        for kind in ['vector', 'potential']:
            saved = torch.load(frozen / ('residual_' + kind + '400.pt'), weights_only=False, map_location='cpu')
            if saved['fit_max_stage'] != 8.25 or not saved['frozen_base_exact']:
                raise ValueError('Frozen field metadata mismatch')
            model = PotentialResidualFlow(base, kind).eval()
            model.residual_net.load_state_dict(saved['residual_net'])
            with torch.no_grad():
                residual = model.residual(1.25, z)
                rn = torch.linalg.vector_norm(residual, dim=1)
                vn = torch.linalg.vector_norm(velocity, dim=1)
                cosine = (residual * velocity).sum(1) / (rn * vn).clamp_min(1e-12)
            fields[kind] = {'residual_norm_median': float(rn.median()), 'residual_to_base_norm_median': float((rn / vn.clamp_min(1e-12)).median()), 'residual_base_cosine_median': float(cosine.median())}
            if not torch.isfinite(residual).all():
                raise ValueError('Nonfinite field')
        results[domain] = {'cells': 1500, 'latent_norm_median': float(torch.linalg.vector_norm(z, dim=1).median()), 'nearest_past_latent_distance_median': float(np.median(nearest)), 'nearest_past_latent_distance_p95': float(np.quantile(nearest, .95)), 'fields': fields}
    for path, sha in pins.items():
        if digest(Path(path)) != sha:
            raise ValueError('Pinned input changed')
    output = {'status': 'completed', 'plan_sha256': digest(run / 'plan.json'), 'results': results, 'new_scoring_batch': False, 'reward_delta': 0, 'scope': plan['scope'], 'limitations': 'Descriptive pointwise latent/field support, not full-panel head-transfer validation or causal explanation of official49.15; only one sampled source/challenge cohort, no independent embryos, no threshold-based promotion.'}
    report.write_text(json.dumps(output, indent=2))
    print(json.dumps(output))


if __name__ == '__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        main()
