"""Own conservative residual architecture; not the Action Matching objective."""
import copy
import torch
from cnf_manifold_flow import rk4_position


class PotentialResidualFlow(torch.nn.Module):
    def __init__(self, base, kind):
        super().__init__()
        if kind not in ('potential', 'vector'):
            raise ValueError('Unknown residual architecture')
        self.base = copy.deepcopy(base).eval()
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.cutoff, self.origin, self.kind = base.cutoff, base.origin, kind
        width, output = (40, 1) if kind == 'potential' else (33, 8)
        self.residual_net = torch.nn.Sequential(torch.nn.Linear(9, 32), torch.nn.Tanh(),
            torch.nn.Linear(32, width), torch.nn.Tanh(), torch.nn.Linear(width, output))
        torch.nn.init.zeros_(self.residual_net[-1].weight)
        torch.nn.init.zeros_(self.residual_net[-1].bias)
        self.residual_enabled = True

    def encode(self, values):
        return self.base.encode(values)

    def residual(self, time, z):
        if self.kind == 'vector':
            return .1 * self.residual_net(torch.cat([z, torch.full_like(z[:, :1], float(time))], 1))
        training_graph = torch.is_grad_enabled()
        # Forecast callers use no_grad. Computing a potential gradient still
        # needs a local graph; training also needs mixed second derivatives.
        with torch.enable_grad():
            position = z if z.requires_grad else z.detach().requires_grad_(True)
            scalar = .1 * self.residual_net(torch.cat([position, torch.full_like(position[:, :1], float(time))], 1))
            gradient = torch.autograd.grad(scalar.sum(), position, create_graph=training_graph)[0]
        return gradient

    def velocity(self, time, z):
        return self.base.velocity(time, z) + self.residual(time, z)

    def trajectory(self, z, times, step=.125):
        if not self.residual_enabled:
            return self.base.trajectory(z, times, step)
        history = [z]
        for a, b in zip(times[:-1], times[1:]):
            z = rk4_position(self.velocity, z, self.cutoff - self.origin + float(a), self.cutoff - self.origin + float(b), step)
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
    for kind in ['vector', 'potential']:
        checkpoint = path.parent / ('residual_' + kind + '400.pt')
        if checkpoint.exists():
            raise ValueError('Never retrain completed arm')
        torch.manual_seed(20261004)
        torch.cuda.reset_peak_memory_stats()
        model = PotentialResidualFlow(base, kind).cuda()
        frozen = {k: v.clone() for k, v in model.base.state_dict().items()}
        trainable = list(model.residual_net.parameters())
        if sum(p.numel() for p in trainable) != 1681:
            raise ValueError('Parameter budget changed')
        optimizer = torch.optim.Adam(trainable, lr=.001)
        rng = torch.Generator().manual_seed(20261004)
        history = []
        for iteration in range(400):
            j = int(torch.randint(3, (1,), generator=rng))
            a = groups[j][torch.randint(len(groups[j]), (64,), generator=rng).cuda()]
            b = groups[j + 1][torch.randint(len(groups[j + 1]), (64,), generator=rng).cuda()]
            optimizer.zero_grad()
            loss = loss_fn(rk4_position(model.velocity, a, float(times[j] - model.origin), float(times[j + 1] - model.origin), step=.125), b)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            if not torch.isfinite(torch.nn.utils.clip_grad_norm_(trainable, 5.)):
                raise ValueError('Nonfinite gradient')
            optimizer.step()
            if (iteration + 1) % 50 == 0:
                record = {'event': 'potential_residual_training', 'kind': kind, 'step': iteration + 1, 'loss': float(loss.detach().cpu())}
                history.append(record)
                with (path.parent / 'events.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
        for key, value in frozen.items():
            torch.testing.assert_close(model.base.state_dict()[key], value, rtol=0, atol=0)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        if peak > 4.5 * 1024 ** 3:
            raise ValueError('GPU memory cap exceeded')
        torch.save({'residual_net': model.residual_net.cpu().state_dict(), 'kind': kind, 'steps': 400,
            'batch_size': 64, 'fit_max_stage': 8.25, 'parameters': 1681, 'training_seed': 20261004,
            'step': .125, 'residual_scale': .1, 'frozen_base_exact': True,
            'base_checkpoint_sha256': digest(packet['base_checkpoint']), 'context_sha256': digest(packet['context']),
            'history': history, 'peak_allocated_bytes': peak, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}, checkpoint)
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
