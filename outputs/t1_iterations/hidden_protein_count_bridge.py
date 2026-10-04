"""Own count/protein resource preflight; no CardamomOT reproduction or scoring."""
import hashlib
import argparse
import json
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from prepare_balanced_atlas import decode_record

HERE = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


class CountProteinFlow(torch.nn.Module):
    """Matched RNA production networks, with fixed finite versus instantaneous protein lag."""
    def __init__(self, genes, kind, seed=20261004):
        super().__init__()
        if kind not in ('instantaneous', 'delayed'):
            raise ValueError('Unknown protein control')
        torch.manual_seed(seed)
        self.kind = kind
        self.left = torch.nn.Parameter(torch.randn(genes, 16) * .01)
        self.right = torch.nn.Parameter(torch.randn(16, genes) * .01)
        self.basal = torch.nn.Parameter(torch.zeros(genes))
        self.decay = torch.nn.Parameter(torch.zeros(genes))
        self.dispersion = torch.nn.Parameter(torch.zeros(genes))

    def forward(self, rna, duration=.25):
        protein = rna.clone()
        dt = duration / max(1, int(np.ceil(duration / (.25 / 16))))
        steps = max(1, int(np.ceil(duration / (.25 / 16))))
        rate = torch.nn.functional.softplus(self.decay) + .01
        for _ in range(steps):
            signal = protein if self.kind == 'delayed' else rna
            production = torch.nn.functional.softplus((torch.log1p(signal) @ self.left) @ self.right + self.basal) + .01
            # Positive exact relaxation for production frozen within each step.
            rna = production / rate + (rna - production / rate) * torch.exp(-rate * dt)
            protein = rna + (protein - rna) * np.exp(-dt)
        return rna

    def mixture_nll(self, source, target, exposure):
        mean = self(source).clamp_min(1e-5)[None, :, :] * exposure[:, None, None]
        theta = (torch.nn.functional.softplus(self.dispersion) + .01)[None, None, :]
        y = target[:, None, :]
        logp = torch.lgamma(y + theta) - torch.lgamma(theta) - torch.lgamma(y + 1)
        logp = logp + theta * (torch.log(theta) - torch.log(theta + mean))
        logp = logp + y * (torch.log(mean) - torch.log(theta + mean))
        return -(torch.logsumexp(logp.sum(-1), dim=1) - np.log(len(source))).mean() / target.shape[1]


def latent_count_delta(initial, predicted, library_reference, projection):
    """Changes in normalized gene means projected through the frozen encoder basis."""
    before = np.log1p(np.asarray(initial, dtype=np.float64) * (10000. / library_reference))
    after = np.log1p(np.asarray(predicted, dtype=np.float64) * (10000. / library_reference))
    return (after - before) @ np.asarray(projection, dtype=np.float64)


class CountBridgeFlow(torch.nn.Module):
    """Count-model residual through the existing encoder and unchanged decoder."""
    def __init__(self, base, count_model, anchor_counts, projection, library_reference, expected_z):
        super().__init__()
        self.base = base
        self.count_model = count_model
        self.cutoff, self.origin = base.cutoff, base.origin
        self.anchor_counts = np.asarray(anchor_counts, dtype=np.float32)
        self.projection = projection
        self.library_reference = float(library_reference)
        self.expected_z = expected_z.clone()
        self.bridge_enabled = True

    def encode(self, values):
        return self.base.encode(values)

    def trajectory(self, z, times, step=.125):
        original = self.base.trajectory(z, times, step)
        if not self.bridge_enabled:
            return original
        torch.testing.assert_close(z, self.expected_z, rtol=0, atol=0)
        if float(times[0]) != 0.:
            raise ValueError('Count bridge requires original cutoff donor initialization')
        history = []
        with torch.no_grad():
            for index, elapsed in enumerate(times):
                if float(elapsed) == 0.:
                    history.append(original[index])
                    continue
                predicted = self.count_model(torch.tensor(self.anchor_counts), float(elapsed)).numpy()
                delta = latent_count_delta(self.anchor_counts, predicted, self.library_reference, self.projection)
                delta /= np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1.)
                history.append(original[index] + torch.tensor(delta, dtype=z.dtype))
        return torch.stack(history)


def train_packet(packet_path):
    packet_path = Path(packet_path).resolve()
    run = packet_path.parent
    if not run.is_relative_to(HERE / 'private'):
        raise ValueError('GPU packet outside evidence root')
    packet = json.loads(packet_path.read_text())
    for path, sha in packet['sha256'].items():
        if digest(path) != sha:
            raise ValueError('GPU packet source/input changed')
    if not torch.cuda.is_available() or torch.__version__ != '2.11.0+cu128':
        raise ValueError('Pinned RTX3060 CUDA runtime required')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.75)
    torch.use_deterministic_algorithms(True)
    with np.load(packet['context']) as context:
        counts, libraries, stages = context['counts'], context['libraries'], context['stages']
        reference = float(context['library_reference'])
    if sorted(np.unique(stages)) != [7.5, 7.75, 8., 8.25]:
        raise ValueError('Past cutoff support changed')
    normalized = counts * (reference / libraries[:, None])
    for kind in ['instantaneous', 'delayed']:
        checkpoint = run / ('protein_' + kind + '400.pt')
        if checkpoint.exists():
            raise ValueError('Never duplicate trained arm')
        torch.cuda.reset_peak_memory_stats()
        model = CountProteinFlow(128, kind).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        rng = np.random.default_rng(20261004)
        history = []
        for iteration in range(400):
            t = [7.5, 7.75, 8.][int(rng.integers(3))]
            a = rng.choice(np.flatnonzero(stages == t), 64, replace=True)
            b = rng.choice(np.flatnonzero(stages == t + .25), 64, replace=True)
            optimizer.zero_grad()
            loss = model.mixture_nll(torch.tensor(normalized[a], dtype=torch.float32, device='cuda'),
                torch.tensor(counts[b], device='cuda'), torch.tensor(libraries[b] / reference, dtype=torch.float32, device='cuda'))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite count loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite count gradients')
            optimizer.step()
            if (iteration + 1) % 50 == 0:
                record = {'event': 'count_protein_training', 'kind': kind, 'step': iteration + 1, 'loss': float(loss.detach().cpu())}
                history.append(record)
                with (run / 'events.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        if peak > 4.5 * 1024 ** 3:
            raise ValueError('GPU memory cap exceeded')
        torch.save({'net': model.cpu().state_dict(), 'kind': kind, 'steps': 400, 'batch_size': 64,
            'fit_max_stage': 8.25, 'parameters': 4480, 'history': history, 'peak_allocated_bytes': peak,
            'context_sha256': digest(packet['context']), 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}, checkpoint)
        del model, optimizer
    for path, sha in packet['sha256'].items():
        if digest(path) != sha:
            raise ValueError('GPU source/input changed during training')
    print(json.dumps({'status': 'completed', 'steps_per_arm': 400, 'trained_device': 'RTX3060'}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='hidden_protein_count_bridge_preflight')
    parser.add_argument('--train-packet')
    args = parser.parse_args()
    if args.train_packet:
        train_packet(args.train_packet)
        return
    entry = json.loads((HERE / 'RESEARCH_HARNESS_MANIFEST.json').read_text())['experiments'][args.experiment]
    run = (HERE / entry['run']).resolve()
    public = (HERE / entry['report']).resolve()
    if not run.is_relative_to(HERE / 'private') or not public.is_relative_to(HERE):
        raise ValueError('Evidence outside project')
    if run.exists() or public.exists():
        raise ValueError('Preserve existing experiment; never rerun')
    run.mkdir(parents=True)
    report = {'status': 'running', 'new_scoring_batch': False, 'score': None, 'reward': 0,
              'scope': 'Past-only count/encoder bridge and GPU resource checks; no scientific batch or author reproduction.'}
    save(public, report)
    try:
        assert HERE.drive.upper() == 'D:'
        prepared = HERE / 'private/associated_prepared_01'
        source = HERE / 'private/extended_atlas'
        encoder_path = HERE / 'private/scfm_projected_past_fold_01/fresh_encoder.npz'
        manifest = json.loads((source / 'manifest.json').read_text())
        hashes = {str(source / r['file']): r['sha256'] for r in manifest['files']}
        prep = json.loads((prepared / 'report.json').read_text())
        for name, key in [('expression.npy', 'expression_sha256'), ('genes.csv', 'genes_sha256'), ('selected_metadata.csv', 'metadata_sha256')]:
            hashes[str(prepared / name)] = prep[key]
        hashes[str(encoder_path)] = digest(encoder_path)
        hashes[str(Path(__file__).resolve())] = digest(__file__)
        hashes[str(HERE / 'prepare_balanced_atlas.py')] = digest(HERE / 'prepare_balanced_atlas.py')
        for path, sha in hashes.items():
            if digest(path) != sha:
                raise ValueError('Input/source changed: ' + path)
        save(run / 'plan.json', {'hashes': hashes, 'cutoff': 8.25, 'genes': 128,
             'selection': 'First128 frozen past-only4096-feature ranking; no target values.',
             'steps_per_arm': 3, 'batch': 64, 'duration': .25,
             'objective': 'NB mixture likelihood over unpaired past target snapshots; target library exposure offsets.',
             'control': 'Same4480parameters; fixed protein decay1/day versus instantaneous protein; no rate fitting/grid.',
             'future_bridge': 'Project count-normalized changes through frozen encoder; full-panel decoder/scorer/guards remain unchanged. Full forecast replay still required before scoring.'})
        metadata = pd.read_csv(prepared / 'selected_metadata.csv')
        past = np.flatnonzero(metadata.numeric_stage.to_numpy() <= 8.25)
        stages = metadata.numeric_stage.to_numpy()[past]
        if sorted(np.unique(stages)) != [7.5, 7.75, 8., 8.25]:
            raise ValueError('Past stage support changed')
        panel = (HERE.parents[1] / 'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
        genes = pd.read_csv(prepared / 'genes.csv')
        unique = genes.loc[~genes.symbol.duplicated(keep=False)]
        lookup = dict(zip(unique.symbol, unique.index))
        with np.load(encoder_path) as encoder:
            features = encoder['features'][:128].copy()
            columns = np.array([lookup[panel[i]] for i in features])
            projection = (encoder['basis'][:, :128] / encoder['scale'][:128]).T.copy()
        raw_x = np.load(prepared / 'expression.npy', mmap_mode='r')
        observed = np.asarray(raw_x[np.ix_(past, columns)], dtype=np.float64)
        index = json.loads((source / 'exprMatrix.json').read_text())
        indexed = sorted([(k, int(v[0]), int(v[1])) for k, v in index.items() if '|' in k], key=lambda x: x[1])
        n = sum(1 for _ in (source / 'meta.tsv').open()) - 1
        rows = metadata.source_row.to_numpy(dtype=int)[past]
        counts = np.empty((len(past), 128), dtype=np.float32)
        with (source / 'exprMatrix.bin').open('rb') as stream:
            for j, col in enumerate(columns):
                key, offset, length = indexed[col]
                if key.split('|')[0] != genes.ensembl.iloc[col]:
                    raise ValueError('Gene order mismatch')
                stream.seek(offset)
                _, vector = decode_record(stream.read(length), n)
                counts[:, j] = vector[rows]
        if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any():
            raise ValueError('Counts not finite nonnegative integers')
        # Derive exposure from observed original counts + saved normalization;
        # never infer counts by exponentiating normalized expression alone.
        ratios = np.divide(10000. * counts, np.expm1(observed), out=np.full_like(observed, np.nan), where=counts > 0)
        libraries = np.rint(np.nanmedian(ratios, axis=1))
        if not np.isfinite(libraries).all() or (libraries <= 0).any():
            raise ValueError('Insufficient count exposure support')
        reconstruction = np.log1p(10000. * counts / libraries[:, None])
        normalization_error = float(np.max(np.abs(reconstruction - observed)))
        if normalization_error > 2e-6:
            raise ValueError('Library-offset normalization reconstruction failed')
        library_reference = float(np.median(libraries))
        normalized_counts = counts * (library_reference / libraries[:, None])
        np.savez_compressed(run / 'past_counts_context.npz', counts=counts, libraries=libraries,
                            stages=stages, columns=columns, panel_features=features,
                            projection=projection, source_rows=rows, library_reference=library_reference)
        neutral = latent_count_delta(normalized_counts[:64], normalized_counts[:64], library_reference, projection)
        np.testing.assert_array_equal(neutral, np.zeros_like(neutral))
        # Verify the bridge reproduces the exact algebra of the frozen encoder.
        changed = normalized_counts[:64] * 1.01
        delta = latent_count_delta(normalized_counts[:64], changed, library_reference, projection)
        direct = (np.log1p(changed * 10000. / library_reference) - np.log1p(normalized_counts[:64] * 10000. / library_reference)) @ projection
        bridge_error = float(np.max(np.abs(delta - direct)))
        if bridge_error > 1e-6:
            raise ValueError('Frozen projection bridge mismatch')
        if not torch.cuda.is_available():
            raise ValueError('CUDA required, no CPU fallback')
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(.75)
        torch.use_deterministic_algorithms(True)
        gpu_results = {}
        for kind in ['instantaneous', 'delayed']:
            torch.cuda.reset_peak_memory_stats()
            model = CountProteinFlow(128, kind).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            rng = np.random.default_rng(20261004)
            started = time.perf_counter()
            losses = []
            for iteration in range(3):
                t = [7.5, 7.75, 8.][iteration]
                a = rng.choice(np.flatnonzero(stages == t), 64, replace=True)
                b = rng.choice(np.flatnonzero(stages == t + .25), 64, replace=True)
                optimizer.zero_grad()
                loss = model.mixture_nll(torch.tensor(normalized_counts[a], dtype=torch.float32, device='cuda'),
                    torch.tensor(counts[b], device='cuda'), torch.tensor(libraries[b] / library_reference, dtype=torch.float32, device='cuda'))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite NB objective')
                loss.backward()
                if not all(torch.isfinite(v.grad).all() for v in model.parameters()):
                    raise ValueError('Nonfinite gradients')
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            torch.cuda.synchronize()
            gpu_results[kind] = {'losses': losses, 'seconds': time.perf_counter() - started,
                'gpu_peak_bytes': torch.cuda.max_memory_allocated(), 'parameters': sum(v.numel() for v in model.parameters())}
            torch.save({'net': model.cpu().state_dict(), 'kind': kind, 'steps': 3, 'scope': 'Preflight only; never reuse as400step trained model.'}, run / (kind + '_preflight.pt'))
            del optimizer, model
        for path, sha in hashes.items():
            if digest(path) != sha:
                raise ValueError('Source/input changed after preflight')
        report.update(status='completed', past_cells=len(past), genes=128,
            normalization_max_error=normalization_error, library_reference=library_reference,
            neutral_bridge_exact=True, projection_bridge_max_error=bridge_error,
            context_sha256=digest(run / 'past_counts_context.npz'), gpu_results=gpu_results,
            next_requirement='Predeclare matched400step trial and frozen full-panel head/replay adapter; resource checks do not establish predictive improvement.')
    except Exception as exc:
        (run / 'traceback.txt').write_text(traceback.format_exc())
        report.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
    save(public, report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
