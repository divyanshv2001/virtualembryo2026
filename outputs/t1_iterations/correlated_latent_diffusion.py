"""Own low-rank SDE covariance control; no author solver code or model copied."""
import copy
import math
import torch
from cnf_manifold_flow import rk4_position


class CorrelatedDiffusionFlow(torch.nn.Module):
    def __init__(self, base, kind, seed=20261004):
        super().__init__()
        if kind not in ('diagonal', 'correlated'):
            raise ValueError('Unknown covariance mechanism')
        self.base = copy.deepcopy(base).eval()
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.cutoff, self.origin = base.cutoff, base.origin
        self.kind, self.seed = kind, seed
        self.noise_net = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.Tanh(), torch.nn.Linear(32, 24))
        torch.nn.init.zeros_(self.noise_net[-1].weight)
        with torch.no_grad():
            self.noise_net[-1].bias[:8].fill_(math.log(.1 / .9))
            self.noise_net[-1].bias[8:].copy_(torch.linspace(-.1, .1, 16))
        self.noise_enabled = True

    def encode(self, x):
        return self.base.encode(x)

    def components(self, z):
        output = self.noise_net(z)
        return .5 * torch.sigmoid(output[:, :8]), .1 * torch.tanh(output[:, 8:]).reshape(-1, 8, 2)

    def covariance(self, z):
        sigma, factor = self.components(z)
        marginal = sigma.square() + factor.square().sum(-1)
        if self.kind == 'diagonal':
            return torch.diag_embed(marginal)
        return torch.diag_embed(sigma.square()) + factor @ factor.transpose(1, 2)

    def evolve(self, z, start, finish, rng, step=.125):
        span = float(finish - start)
        if span < 0:
            raise ValueError('Forward stochastic integration only')
        if span == 0:
            return z
        n = max(1, int(math.ceil(span / step)))
        dt = span / n
        for i in range(n):
            sigma, factor = self.components(z)
            # Both arms draw26standard normals at each cell/step. Equal row
            # variances at identical parameters/state; correlated factor shares
            # two Brownian coordinates, diagonal factor uses16independent ones.
            epsilon = torch.randn(len(z), 26, generator=rng, dtype=z.dtype).to(z.device)
            independent = (factor * epsilon[:, 8:24].reshape(-1, 8, 2)).sum(-1)
            shared = (factor * epsilon[:, None, 24:26]).sum(-1)
            increment = sigma * epsilon[:, :8] + (shared if self.kind == 'correlated' else independent)
            drifted = rk4_position(self.base.velocity, z, start + i * dt, start + (i + 1) * dt, step)
            z = drifted + math.sqrt(dt) * increment
        return z

    def trajectory(self, z, times, step=.125):
        if not self.noise_enabled:
            return self.base.trajectory(z, times, step)
        rng = torch.Generator().manual_seed(self.seed)
        history = [z]
        for a, b in zip(times[:-1], times[1:]):
            z = self.evolve(z, self.cutoff - self.origin + float(a), self.cutoff - self.origin + float(b), rng, step)
            history.append(z)
        return torch.stack(history)


def train_packet(path):
    import json
    import hashlib
    from pathlib import Path
    import numpy as np
    from geomloss import SamplesLoss
    from cnf_manifold_flow import DensityFlowNet
    from graph_kinetic_residual import KineticFlow

    def digest(filename):
        with Path(filename).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()

    path = Path(path).resolve()
    here = Path(__file__).resolve().parent
    if not path.parent.is_relative_to(here / 'private'):
        raise ValueError('Packet outside private D evidence')
    packet = json.loads(path.read_text())
    for filename, sha in packet['sha256'].items():
        if digest(filename) != sha:
            raise ValueError('Source/input changed')
    if not torch.cuda.is_available() or torch.__version__ != '2.11.0+cu128' or '3060' not in torch.cuda.get_device_name(0):
        raise ValueError('Pinned RTX3060 runtime required')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.75)
    torch.use_deterministic_algorithms(True)
    with np.load(packet['context']) as context:
        coordinates, stages = context['coordinates'], context['stages']
    times = np.unique(stages)
    if times.tolist() != [7.5, 7.75, 8., 8.25] or coordinates.shape != (10963, 8):
        raise ValueError('Past-only support changed')
    saved = torch.load(packet['base_checkpoint'], weights_only=False, map_location='cpu')
    if saved['steps'] != 400 or saved['fit_max_stage'] != 8.25 or saved['kind'] != 'none' or not saved['frozen_drift_exact']:
        raise ValueError('Frozen kinetic metadata mismatch')
    state = saved['net']
    drift = DensityFlowNet(state['drift.basis'].numpy(), state['drift.pca_center'].numpy(), 8.25, 7.25)
    base = KineticFlow(drift, state['gene_mean'].numpy(), state['edges'][1].numpy(), state['edges'][0].numpy(), 'none')
    base.load_state_dict(state)
    groups = [torch.tensor(coordinates[stages == t], dtype=torch.float32, device='cuda') for t in times]
    loss_fn = SamplesLoss('sinkhorn', p=2, blur=.05, scaling=.9, backend='tensorized')
    for kind in ['diagonal', 'correlated']:
        checkpoint = path.parent / ('diffusion_' + kind + '400.pt')
        if checkpoint.exists():
            raise ValueError('Never retrain completed arm')
        torch.manual_seed(20261004)
        torch.cuda.reset_peak_memory_stats()
        model = CorrelatedDiffusionFlow(base, kind).cuda()
        frozen = {k: v.clone() for k, v in model.base.state_dict().items()}
        trainable = list(model.noise_net.parameters())
        if sum(p.numel() for p in trainable) != 1080:
            raise ValueError('Parameter budget changed')
        optimizer = torch.optim.Adam(trainable, lr=.001)
        rng = torch.Generator().manual_seed(20261004)
        history = []
        for iteration in range(400):
            j = int(torch.randint(3, (1,), generator=rng))
            a = groups[j][torch.randint(len(groups[j]), (64,), generator=rng).cuda()]
            b = groups[j + 1][torch.randint(len(groups[j + 1]), (64,), generator=rng).cuda()]
            optimizer.zero_grad()
            predicted = model.evolve(a, float(times[j] - model.origin), float(times[j + 1] - model.origin), rng)
            loss = loss_fn(predicted, b)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite diffusion loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, 5.)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite diffusion gradient')
            optimizer.step()
            if (iteration + 1) % 50 == 0:
                record = {'event': 'diffusion_training', 'kind': kind, 'step': iteration + 1, 'loss': float(loss.detach().cpu())}
                history.append(record)
                with (path.parent / 'events.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
        for key, value in model.base.state_dict().items():
            torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        if peak > 4.5 * 1024 ** 3:
            raise ValueError('GPU memory cap exceeded')
        torch.save({'noise_net': model.noise_net.cpu().state_dict(), 'kind': kind, 'steps': 400,
            'batch_size': 64, 'fit_max_stage': 8.25, 'parameters': 1080, 'training_seed': 20261004,
            'split_step': .125, 'normals_per_step': 26, 'frozen_base_exact': True,
            'base_checkpoint_sha256': digest(packet['base_checkpoint']), 'context_sha256': digest(packet['context']),
            'history': history, 'peak_allocated_bytes': peak, 'torch': torch.__version__,
            'gpu': torch.cuda.get_device_name(0)}, checkpoint)
        del model, optimizer, frozen
    for filename, sha in packet['sha256'].items():
        if digest(filename) != sha:
            raise ValueError('Source/input changed during training')
    print(json.dumps({'status': 'completed', 'steps_per_arm': 400, 'device': 'RTX3060'}))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-packet', required=True)
    train_packet(parser.parse_args().train_packet)
