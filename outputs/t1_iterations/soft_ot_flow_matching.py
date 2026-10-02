"""Small past-only coupling-conditioned objective; not a complete scFM reproduction."""
import numpy as np
import torch
from scipy.special import logsumexp


def soft_coupling(left, right, epsilon=.1, max_iterations=300, tolerance=1e-4):
    """Balanced log-domain Sinkhorn on eligible past minibatches only."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1] or not len(left) or not len(right):
        raise ValueError('Invalid past minibatch shapes')
    if not np.isfinite(left).all() or not np.isfinite(right).all() or epsilon <= 0:
        raise ValueError('Invalid coupling inputs')
    cost = np.maximum((left * left).sum(1)[:, None] + (right * right).sum(1)[None, :] - 2 * left @ right.T, 0.)
    scale = max(float(np.median(cost)), 1e-8)
    log_kernel = -cost / (epsilon * scale)
    log_a = np.full(len(left), -np.log(len(left)))
    log_b = np.full(len(right), -np.log(len(right)))
    u, v = np.zeros(len(left)), np.zeros(len(right))
    for iteration in range(max_iterations):
        u = log_a - logsumexp(log_kernel + v[None, :], axis=1)
        v = log_b - logsumexp(log_kernel + u[:, None], axis=0)
        coupling = np.exp(log_kernel + u[:, None] + v[None, :])
        gap = max(float(np.max(np.abs(coupling.sum(1) - np.exp(log_a)))),
                  float(np.max(np.abs(coupling.sum(0) - np.exp(log_b)))))
        if gap <= tolerance:
            break
    if gap > tolerance or not np.isfinite(coupling).all():
        raise ValueError('Past coupling marginal checks failed')
    return coupling / coupling.sum(), {'iterations': iteration + 1, 'max_marginal_error': gap, 'past_cost_scale': scale}


def train_soft_flow(net, z, stages, cutoff, checkpoint, emit, steps=400, batch_size=64, seed=20261002):
    """Stochastic soft pairs avoid barycentric target collapse; actual elapsed days."""
    z, stages = np.asarray(z, dtype=np.float32), np.asarray(stages, dtype=np.float64)
    if len(z) != len(stages) or not np.isfinite(z).all() or not np.isfinite(stages).all() or stages.max() > cutoff:
        raise ValueError('Only finite permitted past latent states accepted')
    times = np.unique(stages)
    if len(times) < 2 or steps != 400 or batch_size != 64:
        raise ValueError('Outside bounded pilot plan')
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    groups = [np.flatnonzero(stages == t) for t in times]
    optimizer = torch.optim.Adam(net.field.parameters(), lr=.001)
    history = []
    for step in range(steps):
        pair = int(rng.integers(len(times) - 1))
        left = z[rng.choice(groups[pair], batch_size, replace=True)]
        right = z[rng.choice(groups[pair + 1], batch_size, replace=True)]
        coupling, audit = soft_coupling(left, right)
        sampled = rng.choice(coupling.size, batch_size, replace=True, p=coupling.ravel())
        start = torch.from_numpy(left[sampled // len(right)])
        finish = torch.from_numpy(right[sampled % len(right)])
        fraction = torch.tensor(rng.random((batch_size, 1)), dtype=torch.float32)
        elapsed = float(times[pair + 1] - times[pair])
        point = start + fraction * (finish - start)
        clock = float(times[pair] - net.origin) + fraction * elapsed
        target = (finish - start) / elapsed
        velocity = net.field(torch.cat([point, clock], dim=1))
        loss = torch.mean((velocity - target) ** 2)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite soft flow loss')
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.field.parameters(), 5.)
        if not torch.isfinite(norm):
            raise ValueError('Nonfinite soft flow gradient')
        optimizer.step()
        if (step + 1) % 50 == 0:
            record = {'step':step + 1,'loss':float(loss.detach()),'start_stage':float(times[pair]),
                      'end_stage':float(times[pair+1]),'elapsed_days':elapsed,**audit}
            history.append(record)
            torch.save({'net':net.state_dict(),'optimizer':optimizer.state_dict(),'numpy_rng':rng.bit_generator.state,
                        'seed':seed,'steps':steps,'history':history,'fit_max_stage':float(stages.max()),
                        'scope':'Local soft OT flow matching adaptation, not full scFM'},checkpoint)
            emit('soft_flow_training_checkpoint',**record)
    net.eval()
    return history
