"""User-requested E10.5 potential-residual progress export; no upload."""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from train_extended_atlas import HERE
from cnf_manifold_flow import DensityFlowNet, rk4_position
from graph_kinetic_residual import KineticFlow
from scalar_potential_residual import PotentialResidualFlow
from run_t1 import digest as path_digest, validate
from iterate import append_event, now

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = 'potential_progress_20261004_02'


def digest(path):
    return path_digest(Path(path))


def load_base(path, encoder, cutoff=9.5):
    saved = torch.load(path, weights_only=False, map_location='cpu')
    if saved['fit_max_stage'] != cutoff or saved['steps'] != 400 or saved['kind'] != 'none' or not saved['frozen_drift_exact']:
        raise ValueError('Released-cutoff kinetic checkpoint mismatch')
    drift = DensityFlowNet(encoder['basis'], encoder['pca_center'], cutoff, 7.25)
    state = saved['net']
    base = KineticFlow(drift, state['gene_mean'].numpy(), state['edges'][1].numpy(), state['edges'][0].numpy(), 'none')
    base.load_state_dict(state)
    return base.eval()


def train(packet_path):
    from geomloss import SamplesLoss
    path = Path(packet_path).resolve()
    if not path.parent.is_relative_to(HERE / 'private'):
        raise ValueError('Packet outside D evidence')
    packet = json.loads(path.read_text())
    for filename, sha in packet['sha256'].items():
        if digest(filename) != sha:
            raise ValueError('Frozen source/input changed')
    if not torch.cuda.is_available() or torch.__version__ != '2.11.0+cu128' or '3060' not in torch.cuda.get_device_name(0):
        raise ValueError('Pinned RTX3060 runtime required')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.75)
    torch.use_deterministic_algorithms(True)
    with np.load(packet['encoder']) as enc:
        encoder = {k: enc[k] for k in enc.files}
    stages = pd.read_csv(packet['metadata']).numeric_stage.to_numpy(float)[encoder['past_rows']]
    times = np.unique(stages)
    if times.tolist() != np.arange(7.5, 9.51, .25).tolist() or len(stages) != 25963:
        raise ValueError('Released past-stage support mismatch')
    torch.manual_seed(20261004)
    model = PotentialResidualFlow(load_base(packet['base'], encoder), 'potential').cuda()
    frozen = {k: v.clone() for k, v in model.base.state_dict().items()}
    parameters = list(model.residual_net.parameters())
    optimizer = torch.optim.Adam(parameters, lr=.001)
    groups = [torch.tensor(encoder['coordinates'][stages == t], dtype=torch.float32, device='cuda') for t in times]
    rng = torch.Generator().manual_seed(20261004)
    loss_fn = SamplesLoss('sinkhorn', p=2, blur=.05, scaling=.9, backend='tensorized')
    for iteration in range(400):
        j = int(torch.randint(len(groups) - 1, (1,), generator=rng))
        a = groups[j][torch.randint(len(groups[j]), (64,), generator=rng).cuda()]
        b = groups[j + 1][torch.randint(len(groups[j + 1]), (64,), generator=rng).cuda()]
        optimizer.zero_grad()
        loss = loss_fn(rk4_position(model.velocity, a, float(times[j] - model.origin), float(times[j + 1] - model.origin), step=.125), b)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        loss.backward()
        if not torch.isfinite(torch.nn.utils.clip_grad_norm_(parameters, 5.)):
            raise ValueError('Nonfinite gradient')
        optimizer.step()
        if (iteration + 1) % 50 == 0:
            append_event(path.parent / 'training_events.jsonl', 'potential_training', step=iteration + 1, loss=float(loss.detach().cpu()))
    for k, v in frozen.items():
        torch.testing.assert_close(model.base.state_dict()[k], v, rtol=0, atol=0)
    peak = torch.cuda.max_memory_allocated()
    if peak > 4.5 * 1024 ** 3:
        raise ValueError('GPU resource cap failed')
    for filename, sha in packet['sha256'].items():
        if digest(filename) != sha:
            raise ValueError('Source/input changed during training')
    torch.save({'residual_net': model.residual_net.cpu().state_dict(), 'fit_max_stage': 9.5,
        'steps': 400, 'seed': 20261004, 'frozen_base_exact': True,
        'base_sha256': digest(packet['base']), 'peak_allocated_bytes': peak,
        'gpu': torch.cuda.get_device_name(0)}, path.parent / 'potential400.pt')


def main():
    import anndata as ad
    from scipy import sparse
    from hurdle_backtest import read_cells
    from full_anchor_slope_forecast import FullAnchorSlopeForecast
    from scnode_resource_preflight import peak_memory
    out = ROOT / 'outputs/t1_submissions' / NAME
    private = HERE / 'private' / NAME
    if out.exists() or private.exists():
        raise ValueError('Never overwrite or duplicate export')
    old = HERE / 'private/kinetic_none_progress_20261004_01'
    data = HERE / 'private/associated_prepared_01'
    old_plan = json.loads((ROOT / 'outputs/t1_submissions/kinetic_none_progress_20261004_01/plan.json').read_text())
    for name, sha in old_plan['source_sha256'].items():
        if name == 'export_anchor_slope_submission.py':
            # This prior orchestration script is not called by this exporter;
            # its disclosure wording changed after the original artifact.
            continue
        if name == 'cnf_density_flow.py' and digest(HERE / name) != sha:
            snapshot = old / name
            if digest(snapshot) != sha:
                raise ValueError('Original density source snapshot changed')
            expected = snapshot.read_text().replace('float(time),dtype=z.dtype)]', 'float(time),dtype=z.dtype,device=z.device)]')
            if (HERE / name).read_text() != expected:
                raise ValueError('Density change exceeds audited CUDA device fix')
            # Only the time tensor device changed; CPU artifact replay below
            # must still pass bit-exact before exporting upgraded forecasts.
            continue
        if digest(HERE / name) != sha:
            raise ValueError('Original export mechanism changed: ' + name)
    prepared = json.loads((data / 'report.json').read_text())
    for filename, key in [('expression.npy', 'expression_sha256'), ('selected_metadata.csv', 'metadata_sha256'), ('genes.csv', 'genes_sha256')]:
        if digest(data / filename) != prepared[key]:
            raise ValueError('Prepared input changed')
    panel_path = ROOT / 'outputs/t1_run/T1__val.genes.txt'
    anchor = ROOT / 'data/E9.5_RNA.h5ad'
    for key, path in [('anchor', anchor), ('panel', panel_path), ('index', ROOT / 'outputs/t1_run/index.json')]:
        if digest(path) != old_plan['input_sha256'][key]:
            raise ValueError('Released observation/panel changed')
    development = HERE / 'SCALAR_POTENTIAL_RESIDUAL_PAST_FOLD_RESULTS.json'
    sources = [Path(__file__), HERE / 'scalar_potential_residual.py', HERE / 'cnf_manifold_flow.py', HERE / 'cnf_density_flow.py', HERE / 'graph_kinetic_residual.py']
    packet = {'encoder': str(old / 'encoder.npz'), 'base': str(old / 'kinetic_none400.pt'), 'metadata': str(data / 'selected_metadata.csv')}
    packet['sha256'] = {str(p): digest(p) for p in sources + [Path(packet[k]) for k in ['encoder', 'base', 'metadata']]}
    out.mkdir(parents=True)
    private.mkdir(parents=True)
    plan = {'created_before_training_utc': now(), 'authorization': 'User requested upgraded submission file; export only.',
        'candidate': 'potential residual', 'observed_cutoff': 9.5, 'forecast_target': 10.5,
        'reuse_scope': 'Exact prior E9.5 encoder/CNF800/kinetics400, refit potential400 on all25963 sampled source cells through9.5; full17057 observedE9.5 head cells,1500 donors.',
        'source_sha256': packet['sha256'], 'development_report_sha256': digest(development),
        'local_development_mean': 52.43846044214097, 'training_steps': 400, 'seed': 20261004,
        'residual_scale': .1, 'step': .125, 'batch_size': 64, 'readiness_gate_passed': False,
        'uploaded': False, 'agent_team_lock_certified': False, 'external_data_disclosure': old_plan['external_data_disclosure'],
        'scope': 'Actual E9.5-to-E10.5 progress forecast; no hidden truth or official score estimate. Potential architecture with Sinkhorn objective, not Action Matching reproduction.'}
    (out / 'plan.json').write_text(json.dumps(plan, indent=2))
    (private / 'GPU_packet.json').write_text(json.dumps(packet, indent=2))
    events = out / 'trajectory.jsonl'
    append_event(events, 'plan_frozen', sha256=digest(out / 'plan.json'))
    python = ROOT / 'outputs/research_workflow/.venv_cuda/Scripts/python.exe'
    with (private / 'GPU_stdout.log').open('wb') as stdout, (private / 'GPU_stderr.log').open('wb') as stderr:
        subprocess.run([str(python), '-u', str(Path(__file__)), '--train-packet', str(private / 'GPU_packet.json')], stdout=stdout, stderr=stderr, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    panel = panel_path.read_text().splitlines()
    with np.load(old / 'encoder.npz') as enc:
        encoder = {k: enc[k] for k in enc.files}
    x = np.load(data / 'expression.npy', mmap_mode='r')
    stages = pd.read_csv(data / 'selected_metadata.csv').numeric_stage.to_numpy(float)
    if stages.max() > 9.5:
        raise ValueError('Past cutoff violation')
    symbols = pd.read_csv(data / 'genes.csv').symbol.fillna('').tolist()
    donors, rows = read_cells(anchor, panel, 1500, 20260928)
    np.testing.assert_array_equal(rows, np.load(old / 'observed_rows.npy'))
    base = load_base(old / 'kinetic_none400.pt', encoder)
    guards = np.load(HERE / 'private/transport_challenge_01/features.npy')
    model = FullAnchorSlopeForecast(x, stages, 9.5, donors, panel, symbols, base,
        encoder['center'], encoder['scale'], encoder['features'], guards, anchor_path=anchor)
    model.configure(.25, .75)
    baseline, _, _ = model.predict(10.5, 'joint', 1., sampling='systematic')
    previous_path = ROOT / 'outputs/t1_submissions/kinetic_none_progress_20261004_01/T1_val__kinetic_none_progress_20261004_01.h5ad'
    if digest(previous_path) != '444923a1c67b4902c426cace1540e83a2001adfb9427b5630029c5ac730b88bd':
        raise ValueError('Original artifact changed')
    previous = ad.read_h5ad(previous_path)
    np.testing.assert_array_equal(baseline, previous.X.toarray())
    del previous, baseline
    append_event(events, 'original_E10_5_kinetic_forecast_exact_replay_passed')
    saved = torch.load(private / 'potential400.pt', weights_only=False, map_location='cpu')
    if saved['fit_max_stage'] != 9.5 or saved['steps'] != 400 or not saved['frozen_base_exact'] or saved['base_sha256'] != digest(old / 'kinetic_none400.pt'):
        raise ValueError('Potential metadata mismatch')
    model.net = PotentialResidualFlow(base, 'potential')
    model.net.residual_net.load_state_dict(saved['residual_net'])
    model.net.eval()
    prediction, indices, audit = model.predict(10.5, 'joint', 1., sampling='systematic')
    replay, replay_indices, _ = model.predict(10.5, 'joint', 1., sampling='systematic')
    np.testing.assert_array_equal(prediction, replay)
    np.testing.assert_array_equal(indices, replay_indices)
    del replay
    protected = np.setdiff1d(np.arange(len(panel)), model.mapped)
    np.testing.assert_array_equal(prediction[:, protected], donors[indices][:, protected])
    before = np.expm1(donors[indices][:, model.mapped].astype(float)).sum(1)
    after = np.expm1(prediction[:, model.mapped].astype(float)).sum(1)
    mass_error = float(np.max(abs(after - before) / np.maximum(before, 1e-12)))
    if mass_error > 1e-5 or peak_memory() >= 16 * 1024 ** 3:
        raise ValueError('Mass/memory guard failed')
    for filename, sha in packet['sha256'].items():
        if digest(filename) != sha:
            raise ValueError('Pinned source/input changed before export')
    artifact = out / ('T1_val__' + NAME + '.h5ad')
    a = ad.AnnData(sparse.csr_matrix(prediction.astype(np.float32)), obs=pd.DataFrame(index=[f'forecast_{i:05d}' for i in range(len(prediction))]), var=pd.DataFrame(index=panel))
    a.uns.update(forecast_target_stage='E10.5', observed_cutoff_stage='E9.5', purpose='User-requested potential progress export; readiness unmet')
    a.write_h5ad(artifact, compression='gzip')
    validation = validate(artifact, panel, json.loads((ROOT / 'outputs/t1_run/index.json').read_text())['T1:val'])
    if not validation['passed']:
        raise ValueError('Submission format failed')
    report = {'status': 'completed', 'artifact': str(artifact), 'format_validation': validation,
        'plan_sha256': digest(out / 'plan.json'), 'forecast_audit': audit, 'mass_error': mass_error,
        'baseline_replay_exact': True, 'candidate_replay_exact': True, 'protected_genes_exact': True,
        'gpu_peak_allocated_bytes': saved['peak_allocated_bytes'], 'host_peak_bytes': peak_memory(),
        'local_development_mean': 52.43846044214097, 'official_score': None,
        'readiness_gate_passed': False, 'uploaded': False, 'agent_team_lock_certified': False}
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    (HERE / 'POTENTIAL_PROGRESS_EXPORT_RESULTS.json').write_text(json.dumps(report, indent=2))
    append_event(events, 'submission_format_passed', sha256=digest(artifact), validation=validation)
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--train-packet')
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.train_packet:
            train(args.train_packet)
        else:
            main()
