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
