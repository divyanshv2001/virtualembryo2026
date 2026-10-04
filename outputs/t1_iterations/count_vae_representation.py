"""Own matched observation-model VAE probe, not an scVI reproduction."""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class ObservationVAE(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind not in ('count_nb', 'log_gaussian'):
            raise ValueError('Unknown observation model')
        self.kind = kind
        self.encoder = torch.nn.Sequential(torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 16))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.ReLU(), torch.nn.Linear(64, 128))
        self.observation_scale = torch.nn.Parameter(torch.zeros(128))

    def encode(self, values):
        mean, logvar = self.encoder(values).chunk(2, dim=-1)
        return mean, logvar.clamp(-8., 8.)

    def decode_fractions(self, z):
        logits = self.decoder(z)
        # Background category accounts for genes outside the fixed128 subset.
        logits = torch.cat([logits, torch.zeros_like(logits[:, :1])], dim=1)
        return torch.softmax(logits, dim=1)[:, :128]

    def loss(self, counts, libraries, epsilon):
        observed = torch.log1p(10000. * counts / libraries[:, None])
        mean, logvar = self.encode(observed)
        z = mean + torch.exp(.5 * logvar) * epsilon
        fractions = self.decode_fractions(z).clamp_min(1e-9)
        scale = torch.nn.functional.softplus(self.observation_scale) + .01
        if self.kind == 'count_nb':
            mu = libraries[:, None] * fractions
            logp = torch.lgamma(counts + scale) - torch.lgamma(scale) - torch.lgamma(counts + 1)
            logp = logp + scale * (torch.log(scale) - torch.log(scale + mu))
            logp = logp + counts * (torch.log(mu) - torch.log(scale + mu))
            reconstruction = -logp.mean()
        else:
            predicted = torch.log1p(10000. * fractions)
            reconstruction = (.5 * ((observed - predicted) / scale).square() + torch.log(scale)).mean()
        kl = .5 * (mean.square() + logvar.exp() - logvar - 1.).sum(1).mean()
        return reconstruction + .01 * kl


class LatentDynamics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.field = torch.nn.Sequential(torch.nn.Linear(9, 32), torch.nn.Tanh(),
            torch.nn.Linear(32, 32), torch.nn.Tanh(), torch.nn.Linear(32, 8))
        torch.nn.init.zeros_(self.field[-1].weight)
        torch.nn.init.zeros_(self.field[-1].bias)

    def velocity(self, time, z):
        t = torch.full((len(z), 1), (float(time) - 7.5) / .75, dtype=z.dtype, device=z.device)
        return self.field(torch.cat([z, t], dim=1))


def train_packet(path):
    from cnf_manifold_flow import rk4_position
    from geomloss import SamplesLoss
    path = Path(path).resolve()
    if not path.parent.is_relative_to(HERE / 'private'):
        raise ValueError('GPU packet outside evidence')
    packet = json.loads(path.read_text())
    for filename, sha in packet['sha256'].items():
        if digest(filename) != sha:
            raise ValueError('GPU source/input changed')
    if not torch.cuda.is_available() or torch.__version__ != '2.11.0+cu128':
        raise ValueError('Pinned RTX3060 runtime required')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.75)
    torch.use_deterministic_algorithms(True)
    with np.load(packet['context']) as data:
        counts, libraries, stages = data['counts'], data['libraries'], data['stages']
    times = np.unique(stages)
    if times.tolist() != [7.5, 7.75, 8., 8.25]:
        raise ValueError('Past support changed')
    x = torch.tensor(counts, device='cuda')
    exposure = torch.tensor(libraries, dtype=torch.float32, device='cuda')
    observed = torch.log1p(10000. * x / exposure[:, None])
    objective = SamplesLoss('sinkhorn', p=2, blur=.05, scaling=.9, backend='tensorized')
    seed = int(packet.get('training_seed', 20261004))
    kinds = packet.get('vae_kinds', ['log_gaussian', 'count_nb'])
    if seed not in (20261004, 20261005) or kinds not in (['log_gaussian', 'count_nb'], ['log_gaussian']):
        raise ValueError('Undeclared seed or model selection')
    for kind in kinds:
        checkpoint = path.parent / ('vae_' + kind + '400.pt')
        if checkpoint.exists():
            raise ValueError('Never retrain completed arm')
        torch.manual_seed(seed)
        torch.cuda.reset_peak_memory_stats()
        model = ObservationVAE(kind).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        rng = torch.Generator().manual_seed(seed)
        history = []
        def record(phase, iteration, loss):
            if (iteration + 1) % 50 == 0:
                item = {'event': 'vae_training', 'kind': kind, 'phase': phase, 'step': iteration + 1, 'loss': float(loss.detach().cpu())}
                history.append(item)
                with (path.parent / 'events.jsonl').open('a') as stream:
                    stream.write(json.dumps(item) + '\n')
        for iteration in range(400):
            rows = torch.randint(len(x), (64,), generator=rng).cuda()
            epsilon = torch.randn(64, 8, generator=rng).cuda()
            optimizer.zero_grad()
            loss = model.loss(x[rows], exposure[rows], epsilon)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite VAE loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite VAE gradient')
            optimizer.step()
            record('representation', iteration, loss)
        model.eval()
        with torch.no_grad():
            coordinates = torch.cat([model.encode(block)[0] for block in observed.split(256)])
            center = coordinates.mean(0)
            scale = coordinates.std(0, unbiased=False).clamp_min(.1)
            coordinates = (coordinates - center) / scale
        groups = [coordinates[torch.tensor(stages == t, device='cuda')] for t in times]
        torch.manual_seed(seed)
        dynamics = LatentDynamics().cuda()
        optimizer = torch.optim.Adam(dynamics.parameters(), lr=.001)
        rng = torch.Generator().manual_seed(seed)
        for iteration in range(400):
            j = int(torch.randint(3, (1,), generator=rng))
            a = groups[j][torch.randint(len(groups[j]), (64,), generator=rng).cuda()]
            b = groups[j + 1][torch.randint(len(groups[j + 1]), (64,), generator=rng).cuda()]
            optimizer.zero_grad()
            pred = rk4_position(dynamics.velocity, a, float(times[j]), float(times[j + 1]), step=.125)
            loss = objective(pred, b)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite latent OT loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(dynamics.parameters(), 5.)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite latent OT gradient')
            optimizer.step()
            record('latent_OT', iteration, loss)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        if peak > 4.5 * 1024 ** 3:
            raise ValueError('GPU memory cap exceeded')
        torch.save({'vae': model.cpu().state_dict(), 'dynamics': dynamics.cpu().state_dict(),
            'center': center.cpu(), 'scale': scale.cpu(), 'kind': kind, 'steps': 400,
            'dynamics_steps': 400, 'batch_size': 64, 'fit_max_stage': 8.25, 'training_seed': seed,
            'vae_parameters': sum(p.numel() for p in model.parameters()),
            'dynamics_parameters': sum(p.numel() for p in dynamics.parameters()),
            'context_sha256': digest(packet['context']), 'history': history,
            'peak_allocated_bytes': peak, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}, checkpoint)
        del model, dynamics, optimizer
    for filename, sha in packet['sha256'].items():
        if digest(filename) != sha:
            raise ValueError('GPU source/input changed during fit')


class VAELatentBridge(torch.nn.Module):
    def __init__(self, base, model, dynamics, center, scale, anchor_values, projection, expected_z):
        super().__init__()
        self.base, self.model, self.dynamics = base, model, dynamics
        self.cutoff, self.origin = base.cutoff, base.origin
        self.center, self.scale = center, scale
        self.anchor_values = torch.tensor(anchor_values, dtype=torch.float32)
        self.projection = torch.tensor(projection, dtype=torch.float32)
        self.expected_z = expected_z.clone()
        self.bridge_enabled = True

    def encode(self, values):
        return self.base.encode(values)

    def trajectory(self, z, times, step=.125):
        from cnf_manifold_flow import rk4_position
        original = self.base.trajectory(z, times, step)
        if not self.bridge_enabled:
            return original
        torch.testing.assert_close(z, self.expected_z, rtol=0, atol=0)
        if float(times[0]) != 0.:
            raise ValueError('Cutoff donor initialization required')
        with torch.no_grad():
            posterior = self.model.encode(self.anchor_values)[0]
            latent = (posterior - self.center) / self.scale
            initial = torch.log1p(10000. * self.model.decode_fractions(posterior))
            history = [original[0]]
            for i, (a, b) in enumerate(zip(times[:-1], times[1:]), start=1):
                latent = rk4_position(self.dynamics.velocity, latent, self.cutoff + float(a), self.cutoff + float(b), step)
                future = torch.log1p(10000. * self.model.decode_fractions(latent * self.scale + self.center))
                delta = (future - initial) @ self.projection
                delta = delta / torch.linalg.vector_norm(delta, dim=1, keepdim=True).clamp_min(1.)
                history.append(original[i] + delta)
        return torch.stack(history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='count_vae_representation_preflight')
    parser.add_argument('--train-packet')
    args = parser.parse_args()
    if args.train_packet:
        train_packet(args.train_packet)
        return
    entry = json.loads((HERE / 'RESEARCH_HARNESS_MANIFEST.json').read_text())['experiments'][args.experiment]
    run, public = (HERE / entry['run']).resolve(), (HERE / entry['report']).resolve()
    if not run.is_relative_to(HERE / 'private') or not public.is_relative_to(HERE):
        raise ValueError('Evidence outside project')
    if run.exists() or public.exists():
        raise ValueError('Never duplicate preflight')
    run.mkdir(parents=True)
    report = {'status': 'running', 'new_scoring_batch': False, 'score': None, 'reward': 0}
    public.write_text(json.dumps(report, indent=2))
    try:
        context = HERE / 'private/hidden_protein_count_bridge_repair_01/past_counts_context.npz'
        expected = '49ac69cc7d42c6c8238d704d3e4b4c1207a458d71081aa110e40474e28175956'
        if digest(context) != expected:
            raise ValueError('Frozen actual count context changed')
        source_sha = digest(__file__)
        plan = {'context_sha256': expected, 'source_sha256': source_sha, 'fit_max_stage': 8.25,
                'rank': 8, 'width': 64, 'genes': 128, 'batch': 64, 'steps_per_arm': 3,
                'matched_control': 'Same architecture,128observation parameters, pairedbatch/epsilon; NB counts vs Gaussian lognormalized likelihood, same KL.01.',
                'scope': 'Resource/finite-gradient and representation encode/decode checks only; not scVI author implementation or predictive validation.'}
        (run / 'plan.json').write_text(json.dumps(plan, indent=2))
        with np.load(context) as data:
            counts, libraries, stages = data['counts'], data['libraries'], data['stages']
        if sorted(np.unique(stages)) != [7.5, 7.75, 8., 8.25]:
            raise ValueError('Past-only support changed')
        if not torch.cuda.is_available() or torch.__version__ != '2.11.0+cu128':
            raise ValueError('Pinned CUDA required; no CPU fallback')
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(.75)
        torch.use_deterministic_algorithms(True)
        results = {}
        for kind in ['log_gaussian', 'count_nb']:
            torch.manual_seed(20261004)
            torch.cuda.reset_peak_memory_stats()
            model = ObservationVAE(kind).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            rng = torch.Generator().manual_seed(20261004)
            losses = []
            started = time.perf_counter()
            for _ in range(3):
                rows = torch.randint(len(counts), (64,), generator=rng).numpy()
                x = torch.tensor(counts[rows], device='cuda')
                exposure = torch.tensor(libraries[rows], dtype=torch.float32, device='cuda')
                epsilon = torch.randn(64, 8, generator=rng).cuda()
                optimizer.zero_grad()
                loss = model.loss(x, exposure, epsilon)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite VAE objective')
                loss.backward()
                if not all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters()):
                    raise ValueError('Nonfinite/missing VAE gradient')
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            with torch.no_grad():
                values = torch.log1p(10000. * x / exposure[:, None])
                mean, _ = model.encode(values)
                fractions = model.decode_fractions(mean)
                if not torch.isfinite(mean).all() or not torch.isfinite(fractions).all() or (fractions < 0).any() or (fractions.sum(1) > 1.).any():
                    raise ValueError('Invalid representation/decoder fractions')
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            if peak > 4.5 * 1024 ** 3:
                raise ValueError('GPU memory cap exceeded')
            results[kind] = {'parameters': sum(v.numel() for v in model.parameters()), 'losses': losses,
                             'seconds': time.perf_counter() - started, 'gpu_peak_bytes': peak}
            model.cpu().eval()
            probe = values.cpu()
            checkpoint = run / (kind + '_preflight.pt')
            torch.save(model.state_dict(), checkpoint)
            clone = ObservationVAE(kind).eval()
            clone.load_state_dict(torch.load(checkpoint, weights_only=True))
            with torch.no_grad():
                torch.testing.assert_close(model.encode(probe)[0], clone.encode(probe)[0], rtol=0, atol=0)
            results[kind]['checkpoint_replay_exact'] = True
            del optimizer, model, clone
        if digest(context) != expected or digest(__file__) != source_sha:
            raise ValueError('Source/input changed after preflight')
        report.update(status='completed', arms=results, past_cells=len(counts), genes=128,
                      scope=plan['scope'], next_requirement='Freeze paired representation/temporal objective and count-to-original-encoder bridge before scoring; neither3step model is trained for prediction.')
    except Exception as exc:
        (run / 'traceback.txt').write_text(traceback.format_exc())
        report.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
    public.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
