"""Bounded scDiffEq-inspired diffusion ablation, not an author reproduction.

Frozen past CNF drift; diagonal noise learned against past snapshot Sinkhorn.
No target data, growth model, spliced counts, or scorer modifications.
"""
import copy
import math
import time
import numpy as np
import torch
from geomloss import SamplesLoss
from cnf_manifold_flow import rk4_position


class DiffusionFlow(torch.nn.Module):
    def __init__(self, drift, kind, seed=20261004):
        super().__init__()
        if kind not in ('constant', 'state'): raise ValueError('Unknown diffusion')
        self.drift = copy.deepcopy(drift).eval()
        for p in self.drift.parameters(): p.requires_grad_(False)
        self.cutoff, self.origin = drift.cutoff, drift.origin
        self.kind, self.seed = kind, seed
        d = len(drift.basis)
        initial = math.log(.1 / .9)  # .5 * sigmoid(logit) = .05
        if kind == 'constant':
            self.logits = torch.nn.Parameter(torch.full((d,), initial))
        else:
            self.diffusion = torch.nn.Sequential(torch.nn.Linear(d, 32), torch.nn.Tanh(), torch.nn.Linear(32, d))
            torch.nn.init.zeros_(self.diffusion[-1].weight)
            torch.nn.init.constant_(self.diffusion[-1].bias, initial)

    def encode(self, x): return self.drift.encode(x)

    def sigma(self, z):
        logits = self.logits.expand_as(z) if self.kind == 'constant' else self.diffusion(z)
        return .5 * torch.sigmoid(logits)

    def evolve(self, z, start, finish, generator, step=.0625, zero=False):
        span = float(finish - start)
        if span < 0: raise ValueError('Forward stochastic integration only')
        n = max(1, int(math.ceil(span / step))); dt = span / n
        for j in range(n):
            sigma = self.sigma(z)
            drifted = rk4_position(self.drift.velocity, z, start + j*dt, start + (j+1)*dt, step=step)
            z = drifted if zero else drifted + math.sqrt(dt)*sigma*torch.randn(z.shape, generator=generator, dtype=z.dtype)
        return z

    def trajectory(self, z, times, step=.0625):
        rng = torch.Generator(device='cpu').manual_seed(self.seed)
        history = [z]
        for a, b in zip(times[:-1], times[1:]):
            z = self.evolve(z, self.cutoff-self.origin+float(a), self.cutoff-self.origin+float(b), rng, step)
            history.append(z)
        return torch.stack(history)


def train_diffusion(model, coordinates, stages, checkpoint, emit, steps=400, batch=64):
    times = np.unique(stages)
    if len(times) < 3 or times.max() > model.cutoff: raise ValueError('Insufficient permitted past support')
    groups = [torch.tensor(coordinates[stages == t], dtype=torch.float32) for t in times]
    if min(map(len, groups)) < batch: raise ValueError('Small past stage')
    rng = torch.Generator(device='cpu').manual_seed(20261004)
    noise = torch.Generator(device='cpu').manual_seed(20261005)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=.001)
    sinkhorn = SamplesLoss('sinkhorn', p=2, blur=.05, scaling=.9, backend='tensorized')
    frozen = {k:v.clone() for k,v in model.drift.state_dict().items()}
    history = []
    for i in range(steps):
        j = int(torch.randint(len(groups)-1, (1,), generator=rng))
        a = groups[j][torch.randint(len(groups[j]), (batch,), generator=rng)]
        b = groups[j+1][torch.randint(len(groups[j+1]), (batch,), generator=rng)]
        forecast = model.evolve(a, float(times[j]-model.origin), float(times[j+1]-model.origin), noise)
        loss = sinkhorn(forecast, b)
        if not torch.isfinite(loss): raise ValueError('Nonfinite diffusion loss')
        opt.zero_grad(); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 5.)
        if not torch.isfinite(norm): raise ValueError('Nonfinite diffusion gradient')
        opt.step()
        if (i+1) % 50 == 0 or i+1 == steps:
            record = {'step':i+1, 'loss':float(loss.detach()), 'sampled_source_stage':float(times[j]), 'sampled_destination_stage':float(times[j+1]), 'sigma_mean':float(model.sigma(a).mean().detach())}
            history.append(record); emit('diffusion_training_checkpoint', diffusion_kind=model.kind, **record)
    for k,v in model.drift.state_dict().items(): torch.testing.assert_close(v, frozen[k], rtol=0, atol=0)
    torch.save({'net':model.state_dict(), 'kind':model.kind, 'steps':steps, 'batch_size':batch, 'fit_max_stage':float(times.max()), 'history':history, 'frozen_drift_exact':True}, checkpoint)
    model.eval()
    return history


def smoke_test(drift):
    started = time.perf_counter()
    torch.manual_seed(20261004)
    model = DiffusionFlow(drift, 'state')
    z = torch.randn(64, len(drift.basis))
    rng = torch.Generator().manual_seed(12)
    a = model.evolve(z, 0., .25, rng)
    loss = SamplesLoss('sinkhorn', p=2, blur=.05, backend='tensorized')(a, z+.1)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    if not all(g is not None and torch.isfinite(g).all() for g in gradients): raise ValueError('Diffusion gradient smoke failed')
    if sum(float(g.abs().sum()) for g in gradients) <= 0: raise ValueError('Zero diffusion gradient')
    if not torch.isfinite(a).all(): raise ValueError('Invalid stochastic integration')
    zero = model.evolve(z, 0., .25, torch.Generator().manual_seed(12), zero=True)
    torch.testing.assert_close(zero, rk4_position(drift.velocity,z,0.,.25,step=.0625), rtol=0, atol=0)
    with torch.no_grad():
        first = model.trajectory(z, torch.tensor([0., .25]))
        torch.testing.assert_close(first,model.trajectory(z,torch.tensor([0.,.25])),rtol=0,atol=0)
    return {'finite_gradient':True,'zero_diffusion_exact_rk4':True,'seeded_replay_exact':True,'seconds':time.perf_counter()-started,'parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
