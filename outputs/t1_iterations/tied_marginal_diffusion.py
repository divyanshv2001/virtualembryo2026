"""Dependence-only residual noise with a frozen cellwise amplitude schedule."""
import copy
import torch
from correlated_latent_diffusion import CorrelatedDiffusionFlow


class TiedMarginalDiffusionFlow(CorrelatedDiffusionFlow):
    def __init__(self, base, reference_noise_net, kind, seed=20261004):
        super().__init__(base, kind, seed)
        self.noise_net = copy.deepcopy(reference_noise_net).eval()
        for parameter in self.noise_net.parameters():
            parameter.requires_grad_(False)
        self.factor_net = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.Tanh(), torch.nn.Linear(32, 16))
        torch.nn.init.zeros_(self.factor_net[-1].weight)
        with torch.no_grad():
            self.factor_net[-1].bias.copy_(torch.linspace(-.1, .1, 16))

    def reference_variance(self, z):
        sigma, factor = super().components(z)
        return sigma.square() + factor.square().sum(-1)

    def components(self, z):
        amplitude = self.reference_variance(z).sqrt()
        loading = .5 * torch.tanh(self.factor_net(z)).reshape(-1, 8, 2)
        # Normalize each row of [I, loading], preserving reference marginal
        # variance at any identical state, including after factor updates.
        sigma = amplitude / (1. + loading.square().sum(-1)).sqrt()
        return sigma, sigma[..., None] * loading


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
    reference = CorrelatedDiffusionFlow(base, 'diagonal')
    saved = torch.load(packet['amplitude_checkpoint'], weights_only=False, map_location='cpu')
    if saved['kind'] != 'diagonal' or saved['fit_max_stage'] != 8.25 or saved['steps'] != 400 or saved['parameters'] != 1080 or saved['base_checkpoint_sha256'] != digest(packet['base_checkpoint']):
        raise ValueError('Frozen amplitude metadata mismatch')
    reference.noise_net.load_state_dict(saved['noise_net'])
    groups = [torch.tensor(coordinates[stages == t], dtype=torch.float32, device='cuda') for t in times]
    loss_fn = SamplesLoss('sinkhorn', p=2, blur=.05, scaling=.9, backend='tensorized')
    for kind in ['diagonal', 'correlated']:
        checkpoint = path.parent / ('tied_' + kind + '400.pt')
        if checkpoint.exists():
            raise ValueError('Never retrain completed arm')
        torch.manual_seed(20261004)
        torch.cuda.reset_peak_memory_stats()
        model = TiedMarginalDiffusionFlow(base, reference.noise_net, kind).cuda()
        frozen = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith('factor_net.')}
        trainable = list(model.factor_net.parameters())
        if sum(p.numel() for p in trainable) != 816:
            raise ValueError('Parameter budget changed')
        optimizer = torch.optim.Adam(trainable, lr=.001)
        rng = torch.Generator().manual_seed(20261004)
        history = []
        for iteration in range(400):
            j = int(torch.randint(3, (1,), generator=rng))
            a = groups[j][torch.randint(len(groups[j]), (64,), generator=rng).cuda()]
            b = groups[j + 1][torch.randint(len(groups[j + 1]), (64,), generator=rng).cuda()]
            optimizer.zero_grad()
            loss = loss_fn(model.evolve(a, float(times[j] - model.origin), float(times[j + 1] - model.origin), rng), b)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            if not torch.isfinite(torch.nn.utils.clip_grad_norm_(trainable, 5.)):
                raise ValueError('Nonfinite gradient')
            optimizer.step()
            if (iteration + 1) % 50 == 0:
                record = {'event': 'tied_diffusion_training', 'kind': kind, 'step': iteration + 1, 'loss': float(loss.detach().cpu())}
                history.append(record)
                with (path.parent / 'events.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
        for key, value in frozen.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        with torch.no_grad():
            probes = torch.cat(groups)
            cov = model.covariance(probes)
            error = (cov.diagonal(dim1=-2, dim2=-1) - model.reference_variance(probes)).abs().max().item()
            if error > 1e-8 or torch.linalg.eigvalsh(cov).min() <= 0:
                raise ValueError('Marginal variance/positive-definiteness gate failed')
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        if peak > 4.5 * 1024 ** 3:
            raise ValueError('GPU memory cap exceeded')
        torch.save({'factor_net': model.factor_net.cpu().state_dict(), 'kind': kind, 'steps': 400,
            'batch_size': 64, 'fit_max_stage': 8.25, 'parameters': 816, 'training_seed': 20261004,
            'split_step': .125, 'normals_per_step': 26, 'frozen_base_amplitude_exact': True,
            'base_checkpoint_sha256': digest(packet['base_checkpoint']), 'amplitude_checkpoint_sha256': digest(packet['amplitude_checkpoint']),
            'context_sha256': digest(packet['context']), 'history': history, 'marginal_variance_max_error': error,
            'peak_allocated_bytes': peak, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}, checkpoint)
        del model, optimizer, frozen, probes, cov
    for filename, sha in packet['sha256'].items():
        if digest(filename) != sha:
            raise ValueError('Source/input changed during training')
    print(json.dumps({'status': 'completed', 'steps_per_arm': 400, 'device': 'RTX3060'}))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-packet', required=True)
    train_packet(parser.parse_args().train_packet)
