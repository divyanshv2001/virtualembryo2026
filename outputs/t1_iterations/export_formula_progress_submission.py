"""Prospective E10.5 T1 artifact from the frozen conservative covariance formula.

This only exports and locally validates a user-requested file. It never uploads.
"""
import json
from collections import Counter

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from threadpoolctl import threadpool_limits

from train_extended_atlas import HERE
from run_t1 import digest, validate
from iterate import now, append_event
from hurdle_backtest import read_cells
from cnf_manifold_flow import DensityFlowNet
from full_anchor_slope_forecast import FullAnchorSlopeForecast
from cnf_covariance_alignment import CovarianceAlignedTrajectory, symmetric_power


def main():
    root = HERE.parents[1]
    name = 'formula_progress_20260930_01'
    out = root / 'outputs/t1_submissions' / name
    private = HERE / 'private' / name
    if out.exists() or private.exists():
        raise ValueError('Preserve prior formula export; use another version')
    prepared = HERE / 'private/associated_prepared_01'
    previous = HERE / 'private/anchorslope_progress_20260929_01'
    dev = HERE / 'CNF_COVARIANCE_ALIGNMENT_RESULTS.json'
    anchor_path = root / 'data/E9.5_RNA.h5ad'
    panel_path = root / 'outputs/t1_run/T1__val.genes.txt'
    index_path = root / 'outputs/t1_run/index.json'
    panel = panel_path.read_text().splitlines()
    source_files = ['export_formula_progress_submission.py', 'cnf_covariance_alignment.py',
                    'full_anchor_slope_forecast.py', 'partial_anchor_forecast.py',
                    'log1p_positive_forecast.py', 'feature_panel_forecast.py',
                    'cnf_manifold_flow.py', 'offline_backtest.py', 'temporary_forecast_cache.py']
    plan = {
        'created_before_fitting_utc': now(),
        'authorization': 'User requested a submission .h5ad after the formula experiment. Export only; no upload.',
        'observed_cutoff': 9.5, 'target': 10.5, 'source_max_stage': 9.5,
        'candidate': 'source/challenge E9.5 8D mean plus covariance .25; full E9.5 anchor slope .25/.75',
        'candidate_selection': 'EB historical proxy screen failed; conservative covariance .25 was strongest eligible narrow E8.5->E9.5 development pilot (55.98883 mean), but its >=60 expansion rule failed. Prospective transfer is unvalidated.',
        'covariance_strength': .25, 'covariance_ridge_fraction': .05,
        'map_eigenvalue_clip': [.5, 2.], 'positive_alpha': .25, 'detection_alpha': .75,
        'donor_seed': 20260928, 'donors': 1500, 'cells_written': 1500,
        'input_sha256': {'anchor': digest(anchor_path), 'panel': digest(panel_path),
                         'index': digest(index_path), 'prepared_report': digest(prepared / 'report.json'),
                         'prior_encoder': digest(previous / 'encoder.npz'),
                         'prior_flow_checkpoint': digest(previous / 'training.pt'),
                         'development_result': digest(dev)},
        'source_sha256': {f: digest(HERE / f) for f in source_files},
        'method': 'Reuse past-only encoder/CNF trained through E9.5 from the prior prospective export; refit full E9.5 challenge anchor conditional slopes and stage-matched source/challenge 8D mean/covariance using only observed E9.5. Transport E9.5 donor states for one day with regularized covariance alignment .25; decoder keeps protected genes and per-cell mapped abundance mass.',
        'limitations': 'No E10.5 truth is used. Reused E8.5->E9.5 development is not an official E10.5 score; local >72/temporal readiness gates remain unmet. External atlas must be disclosed.',
        'external_data_disclosure': 'Development2024 dev201867 public atlas, associated_prepared_01 sample (25,963 cells), through E9.5; no GSE76118 expression used.',
        'official_score': None, 'uploaded': False, 'submissions_allowed': 0,
    }
    out.mkdir(parents=True)
    private.mkdir(parents=True)
    (out / 'plan.json').write_text(json.dumps(plan, indent=2))
    events = out / 'trajectory.jsonl'
    append_event(events, 'export_plan_frozen', sha256=digest(out / 'plan.json'))
    prep_report = json.loads((prepared / 'report.json').read_text())
    for filename, key in [('expression.npy', 'expression_sha256'),
                      ('selected_metadata.csv', 'metadata_sha256'), ('genes.csv', 'genes_sha256')]:
        if digest(prepared / filename) != prep_report[key]:
            raise ValueError('Prepared source changed: ' + filename)
    x = np.load(prepared / 'expression.npy', mmap_mode='r')
    stages = pd.read_csv(prepared / 'selected_metadata.csv').numeric_stage.to_numpy(float)
    if not np.isfinite(stages).all() or stages.max() > 9.5:
        raise ValueError('Source beyond observed cutoff')
    symbols = pd.read_csv(prepared / 'genes.csv').symbol.fillna('').tolist()
    donors, donor_rows = read_cells(anchor_path, panel, 1500, 20260928)
    np.testing.assert_array_equal(donor_rows, np.load(previous / 'observed_rows.npy'))
    np.save(private / 'observed_rows.npy', donor_rows)
    append_event(events, 'observed_inputs_loaded', anchor_cells=len(donors),
                 source_cells=int(len(stages)), maximum_source_stage=float(stages.max()))
    with np.load(previous / 'encoder.npz') as encoder:
        features = encoder['features']; center = encoder['center']; scale = encoder['scale']
        net = DensityFlowNet(encoder['basis'], encoder['pca_center'], 9.5,
                             float(stages[encoder['past_rows']].min())-.25)
    net.load_state_dict(torch.load(previous / 'training.pt', weights_only=False,
                                   map_location='cpu')['net'])
    net.eval()
    guards = np.load(HERE / 'private/transport_challenge_01/features.npy')
    program = FullAnchorSlopeForecast(x, stages, 9.5, donors, panel, symbols, net,
                                      center, scale, features, guards,
                                      anchor_path=anchor_path)
    program.configure(.25, .75)
    counts = Counter(symbols)
    lookup = {s: i for i, s in enumerate(symbols) if s and counts[s] == 1}
    atlas_features = np.array([lookup[panel[i]] for i in features])
    source_rows = np.flatnonzero(stages == 9.5)
    with torch.no_grad():
        source_z = net.encode(torch.tensor(
            (np.asarray(x[np.ix_(source_rows, atlas_features)]) - center) / scale,
            dtype=torch.float32))[0].numpy().astype(np.float64)
    challenge_z = np.empty((program.audit['anchor_calibration_rows'], len(program.zcenter)),
                           dtype=np.float64)
    a = ad.read_h5ad(anchor_path, backed='r')
    try:
        with torch.no_grad():
            for start in range(0, len(challenge_z), 256):
                end = min(start+256, len(challenge_z))
                block = a.X[start:end, features]
                block = block.toarray() if sparse.issparse(block) else np.asarray(block)
                challenge_z[start:end] = net.encode(torch.tensor(
                    (block.astype(np.float32)-center)/scale))[0].numpy()
    finally:
        a.file.close()
    source_mean = source_z.mean(0)
    challenge_mean = challenge_z.mean(0)
    np.testing.assert_allclose(challenge_mean, program.anchor_latent_mean+program.zcenter,
                               atol=1e-6)
    cs = np.cov(source_z, rowvar=False)
    cc = np.cov(challenge_z, rowvar=False)
    ridge = .05*(np.trace(cs)+np.trace(cc))/(2*cs.shape[0])
    cs += ridge*np.eye(len(source_mean))
    cc += ridge*np.eye(len(source_mean))
    root_c = symmetric_power(cc, .5)
    inverse_c = symmetric_power(cc, -.5)
    full_map = inverse_c @ symmetric_power(root_c@cs@root_c, .5) @ inverse_c
    eigenvalues, vectors = np.linalg.eigh((full_map+full_map.T)/2)
    clipped = np.clip(eigenvalues, .5, 2.)
    full_map = (vectors*clipped)@vectors.T
    transform = np.eye(len(source_mean))+.25*(full_map-np.eye(len(source_mean)))
    np.savez_compressed(private / 'alignment.npz', source_mean=source_mean,
                        challenge_mean=challenge_mean, source_cov=cs,
                        challenge_cov=cc, map_eigenvalues=eigenvalues,
                        clipped_eigenvalues=clipped, transform=transform)
    append_event(events, 'past_only_alignment_fitted', source_stage_rows=len(source_rows),
                 challenge_stage_rows=len(challenge_z), ridge=float(ridge),
                 map_eigenvalues=eigenvalues.tolist(),
                 alignment_sha256=digest(private / 'alignment.npz'))
    program.net = CovarianceAlignedTrajectory(net, source_mean, challenge_mean, transform)
    pred, indices, audit = program.predict(10.5, 'joint', 1., sampling='systematic')
    np.save(private / 'donor_indices.npy', indices)
    protected = np.setdiff1d(np.arange(len(panel)), program.mapped)
    exact = bool(np.array_equal(pred[:, protected], donors[indices][:, protected]))
    before = np.expm1(donors[indices][:, program.mapped].astype(float)).sum(1)
    after = np.expm1(pred[:, program.mapped].astype(float)).sum(1)
    error = float(np.max(np.abs(after-before)/np.maximum(before, 1e-12)))
    if not exact or error > 1e-5 or not np.isfinite(pred).all() or (pred < 0).any():
        raise ValueError('Prospective forecast guard failed')
    append_event(events, 'E10_5_forecast_frozen', shape=list(pred.shape),
                 protected_genes_exact=exact, mapped_mass_max_relative_error=error,
                 audit=audit)
    artifact = out / f'T1_val__{name}.h5ad'
    ann = ad.AnnData(sparse.csr_matrix(pred.astype(np.float32)),
                     obs=pd.DataFrame(index=[f'forecast_{i:05d}' for i in range(len(pred))]),
                     var=pd.DataFrame(index=panel))
    ann.uns['forecast_target_stage'] = 'E10.5'
    ann.uns['observed_cutoff_stage'] = 'E9.5'
    ann.uns['purpose'] = 'User-requested prospective progress export; >72 readiness unmet'
    ann.write_h5ad(artifact, compression='gzip')
    del ann, pred
    specification = json.loads(index_path.read_text())['T1:val']
    validation = validate(artifact, panel, specification)
    if not validation['passed']:
        raise ValueError('Submission format validation failed')
    report = {'artifact': artifact.name, 'artifact_sha256': digest(artifact),
              'plan_sha256': digest(out / 'plan.json'), 'format_validation': validation,
              'forecast_audit': audit, 'protected_genes_exact': exact,
              'mapped_mass_max_relative_error': error,
              'alignment_sha256': digest(private / 'alignment.npz'),
              'development_local_mean': 55.988830, 'development_scope': 'E8.5->E9.5 reused panels',
              'readiness_gate_passed': False, 'official_score': None, 'uploaded': False,
              'external_data_disclosure': plan['external_data_disclosure'],
              'evidence_scope': 'Trajectory describes this export only, not the complete research workflow.'}
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    (out / 'README.md').write_text(
        f'# Prospective T1 E10.5 progress file\n\nUpload `{artifact.name}` for `T1:val` if you choose to spend an official attempt. '
        'This is a new E10.5 forecast generated from observed E9.5 and the public atlas through E9.5, not an E9.5 backtest. '
        'It has 1,500 cells, 32,285 genes in exact order, finite nonnegative float32 expression, and no coordinates.\n\n'
        'The selected mechanism scored 55.99 only on reused local E8.5-to-E9.5 development panels. '
        'The >72 local readiness gate was not met; that value does not predict its official E10.5 score. '
        'No upload was performed. Disclose the Development2024 dev201867 atlas if submitting. '
        'The trajectory is genuine for this export, but is not the complete Agent Team research trajectory.\n')
    append_event(events, 'submission_artifact_format_validated',
                 artifact_sha256=report['artifact_sha256'], validation=validation)
    print(json.dumps({'artifact': str(artifact), 'validation': validation,
                      'readiness_gate_passed': False, 'uploaded': False}))


if __name__ == '__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        main()
