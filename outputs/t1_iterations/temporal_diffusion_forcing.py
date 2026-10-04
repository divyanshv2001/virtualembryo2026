"""Own frozen-PCA causal denoising adaptation, not a CellPace reproduction."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class TemporalDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Linear(8, 32)
        self.level = torch.nn.Linear(1, 32)
        self.stage = torch.nn.Linear(1, 32)
        self.block = torch.nn.TransformerEncoderLayer(32, 4, 64, dropout=0., batch_first=True)
        self.output = torch.nn.Linear(32, 8)

    def forward(self, x, levels, stages):
        h = self.input(x) + self.level(levels[..., None] / 99.)
        h = h + self.stage((stages[..., None] - 7.5) / .75)
        mask = torch.ones(3, 3, device=x.device, dtype=torch.bool).triu(1)
        return self.output(self.block(h, src_mask=mask))


def schedule(device='cpu'):
    return torch.cumprod(1. - torch.linspace(.0001, .1, 100, device=device), 0)


def train_packet(path):
    path = Path(path).resolve()
    if not path.parent.is_relative_to(HERE / 'private'):
        raise ValueError('GPU packet outside D evidence')
    packet = json.loads(path.read_text())
    for filename, sha in packet['sha256'].items():
        if digest(filename) != sha:
            raise ValueError('Source/input changed')
    if not torch.cuda.is_available() or torch.__version__ != '2.11.0+cu128':
        raise ValueError('Pinned RTX3060 runtime required')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.75)
    torch.use_deterministic_algorithms(True)
    with np.load(packet['context']) as context:
        coordinates, stages = context['coordinates'], context['stages']
    times = np.unique(stages)
    if times.tolist() != [7.5, 7.75, 8., 8.25]:
        raise ValueError('Past-only support changed')
    groups = [torch.tensor(coordinates[stages == t], dtype=torch.float32, device='cuda') for t in times]
    alpha = schedule('cuda')
    for kind in ['conditional', 'forcing']:
        checkpoint = path.parent / ('temporal_' + kind + '400.pt')
        if checkpoint.exists():
            raise ValueError('Never retrain completed arm')
        torch.manual_seed(20261004)
        torch.cuda.reset_peak_memory_stats()
        model = TemporalDenoiser().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        rng = torch.Generator().manual_seed(20261004)
        history = []
        for iteration in range(400):
            start = int(torch.randint(2, (1,), generator=rng))
            ids = [torch.randint(len(groups[start + j]), (64,), generator=rng).cuda() for j in range(3)]
            clean = torch.stack([groups[start + j][ids[j]] for j in range(3)], dim=1)
            levels = torch.randint(100, (64, 3), generator=rng).cuda()
            noise = torch.randn(64, 3, 8, generator=rng).cuda()
            if kind == 'conditional':
                levels[:, :2] = -1
            a = alpha[levels.clamp_min(0)]
            a = torch.where(levels < 0, torch.ones_like(a), a)
            noisy = a[..., None].sqrt() * clean + (1. - a[..., None]).sqrt() * noise
            positions = torch.tensor(times[start:start + 3], dtype=torch.float32, device='cuda').expand(64, -1)
            optimizer.zero_grad()
            loss = (model(noisy, levels, positions)[:, -1] - clean[:, -1]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite temporal loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite temporal gradient')
            optimizer.step()
            if (iteration + 1) % 50 == 0:
                record = {'event': 'temporal_training', 'kind': kind, 'step': iteration + 1, 'loss': float(loss.detach().cpu())}
                history.append(record)
                with (path.parent / 'events.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        if peak > 4.5 * 1024 ** 3:
            raise ValueError('GPU memory cap exceeded')
        torch.save({'net': model.cpu().state_dict(), 'kind': kind, 'steps': 400, 'batch_size': 64,
            'fit_max_stage': 8.25, 'parameters': sum(p.numel() for p in model.parameters()),
            'history': history, 'peak_allocated_bytes': peak, 'context_sha256': digest(packet['context']),
            'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}, checkpoint)
        del model, optimizer
    for filename, sha in packet['sha256'].items():
        if digest(filename) != sha:
            raise ValueError('Source/input changed during training')
    print(json.dumps({'status': 'completed', 'steps_per_arm': 400, 'trained_device': 'RTX3060'}))


class TemporalBridgeFlow(torch.nn.Module):
    def __init__(self, base, model, earlier_context, expected_z):
        super().__init__()
        self.base, self.model = base, model
        self.cutoff, self.origin = base.cutoff, base.origin
        self.earlier_context = torch.tensor(earlier_context, dtype=torch.float32)
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
            raise ValueError('Expected cutoff initialization')
        rng = torch.Generator().manual_seed(20261004)
        earlier = self.earlier_context[torch.randint(len(self.earlier_context), (len(z),), generator=rng)]
        context = torch.stack([earlier, z], dim=1)
        alpha = schedule()
        reverse = np.linspace(99, 0, 20, dtype=int).tolist() + [-1]
        history = [original[0]]
        previous = 0.
        with torch.no_grad():
            for index, elapsed in enumerate(times[1:], start=1):
                elapsed = float(elapsed)
                steps = int(round((elapsed - previous) / .25))
                if abs(steps * .25 - (elapsed - previous)) > 1e-6 or steps < 1:
                    raise ValueError('Only predeclared quarter-day rollout supported')
                for quarter in range(steps):
                    end = self.cutoff + previous + (quarter + 1) * .25
                    positions = torch.tensor([end - .5, end - .25, end], dtype=z.dtype).expand(len(z), -1)
                    future = torch.randn(len(z), 8, generator=rng)
                    for current, following in zip(reverse[:-1], reverse[1:]):
                        levels = torch.tensor([-1, -1, current]).expand(len(z), -1)
                        inputs = torch.cat([context, future[:, None]], dim=1)
                        x0 = self.model(inputs, levels, positions)[:, -1].clamp(-8., 8.)
                        noise = (future - alpha[current].sqrt() * x0) / (1. - alpha[current]).sqrt()
                        a_next = alpha[following] if following >= 0 else torch.tensor(1.)
                        future = a_next.sqrt() * x0 + (1. - a_next).sqrt() * noise
                    context = torch.stack([context[:, -1], future], dim=1)
                delta = future - original[index]
                delta = delta / torch.linalg.vector_norm(delta, dim=1, keepdim=True).clamp_min(1.)
                history.append(original[index] + delta)
                previous = elapsed
        return torch.stack(history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-packet', required=True)
    train_packet(parser.parse_args().train_packet)
