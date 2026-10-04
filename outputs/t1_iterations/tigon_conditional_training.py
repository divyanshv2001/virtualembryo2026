"""Fixed paired conditional TIGON training and identity-preserving decoding."""
import copy
import hashlib
import time
import numpy as np
import torch
from tigon_conditional_core import ConditionalUOT, objective, integrate, conditional_weights
from scnode_resource_preflight import peak_memory

SEED = 20261002
LIMIT = 16 * 1024**3


def train_pair(coordinates, stages, cutoff, run, emit):
    times = np.unique(stages)
    if times.tolist() != [7.5, 7.75] or cutoff != 7.75:
        raise ValueError('Undeclared training scope')
    groups = [torch.tensor(coordinates[stages == t], dtype=torch.float32) for t in times]
    times = (times - times[0]).tolist()
    torch.manual_seed(SEED)
    initial = copy.deepcopy(ConditionalUOT().state_dict())
    torch.save(initial, run/'tigon_shared_initial.pt')
    flows = {}; matched = None
    for enabled in [False, True]:
        name = 'conditional_tigon_growth_' + ('enabled' if enabled else 'disabled')
        model = ConditionalUOT(growth=enabled); model.load_state_dict(initial)
        generator = torch.Generator().manual_seed(SEED)
        optimizer = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=.01)
        history = []; stream = hashlib.sha256()
        for step in range(200):
            # Same sampler as the passed numerical pilot, now all past KDE centers.
            queries = [g[torch.randperm(len(g), generator=generator)[:32]] +
                       torch.randn(32, 8, generator=generator)*np.sqrt(.02) for g in groups[1:]]
            start = groups[0][torch.randperm(len(groups[0]), generator=generator)[:32]] + torch.randn(32, 8, generator=generator)*np.sqrt(.02)
            for value in queries + [start]: stream.update(value.numpy().tobytes())
            started = time.perf_counter()
            loss = objective(model, groups, times, queries, start)
            if not torch.isfinite(loss): raise ValueError('Nonfinite TIGON loss')
            optimizer.zero_grad(); loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite TIGON gradient')
            memory = peak_memory()
            if memory >= LIMIT: raise ValueError('Real-past 16GiB resource gate failed')
            if step == 0:
                emit('real_past_resource_gate_passed', candidate=name, centers=[len(g) for g in groups],
                     peak_process_working_set_bytes=memory, forward_backward_seconds=time.perf_counter()-started,
                     queries=32, steps=200, density_chunk=256)
            optimizer.step()
            if (step+1)%25 == 0:
                record = {'candidate':name, 'step':step+1, 'loss':float(loss.detach()), 'peak_process_working_set_bytes':memory}
                history.append(record); emit('tigon_training_checkpoint', **record)
                torch.save({'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'generator':generator.get_state(),
                            'history':history, 'seed':SEED, 'fit_max_stage':cutoff, 'steps':200,
                            'query_noise_variance':.02}, run/(name+'.pt'))
        signature = stream.hexdigest()
        if matched is not None and matched != signature: raise ValueError('Paired TIGON streams differ')
        matched = signature; emit('matched_tigon_stream', candidate=name, draw_sha256=signature)
        flows[name] = copy.deepcopy(model).eval()
    return flows


def systematic_indices(weights, seed=20260928):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or not len(weights) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Invalid particle weights')
    weights = weights / weights.sum()
    offset = np.random.default_rng(seed).random()
    positions = (np.arange(len(weights)) + offset) / len(weights)
    cumulative = np.cumsum(weights); cumulative[-1] = 1.
    return np.searchsorted(cumulative, positions, side='right')


class FrozenTrajectory:
    def __init__(self, encoder, z0, z1): self.encoder, self.z0, self.z1 = encoder, z0, z1
    def encode(self, values): return self.encoder.encode(values)
    def trajectory(self, z, times):
        torch.testing.assert_close(z, self.z0, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(times, torch.tensor([0., .25]))
        return torch.stack([z, self.z1])


def forecast(reference, model, donor_rows, target, run, name):
    if target != 8. or reference.cutoff != 7.75: raise ValueError('Undeclared forecast interval')
    with torch.no_grad():
        z0 = reference.net.encode(torch.tensor((reference.donors[:,reference.features]-reference.center)/reference.scale))[0]
    # Preserve exact divergence while releasing each particle block's autograd graph.
    finals = []; logs = []
    for start in range(0, len(z0), 32):
        z, logw, _, _ = integrate(model, z0[start:start+32].clone(), .25, .5)
        finals.append(z.detach()); logs.append(logw.detach())
    z1 = torch.cat(finals); logw = torch.cat(logs)
    weights, ess = conditional_weights(logw)
    if not torch.isfinite(z1).all() or not torch.isfinite(logw).all() or not 1 <= float(ess) <= len(z0)+1e-3:
        raise ValueError('Invalid conditional forecast/ESS')
    if not model.growth_enabled:
        torch.testing.assert_close(weights, torch.full_like(weights, 1/len(weights)))
    indices = systematic_indices(weights.numpy())
    if not model.growth_enabled: np.testing.assert_array_equal(indices, np.arange(len(indices)))
    np.savez_compressed(run/(name+'_particles.npz'), z0=z0.numpy(), z1=z1.numpy(),
                        logw=logw.numpy(), weights=weights.numpy(), indices=indices, donor_rows=donor_rows[indices])
    decoder = copy.copy(reference)
    decoder.donors = reference.donors[indices].copy()
    decoder.audit = dict(reference.audit)
    decoder.net = FrozenTrajectory(reference.net, z0[indices], z1[indices])
    pred, decoder_ids, audit = decoder.predict(target, 'joint', 1., sampling='systematic')
    np.testing.assert_array_equal(decoder_ids, np.arange(len(indices)))
    audit.update(method='TIGON-inspired conditional particles with unchanged past anchor heads',
                 fit_max_stage=7.75, growth_enabled=model.growth_enabled, endpoint_masses=[1.,1.],
                 conditional_weight_ess=float(ess), selected_unique_donors=len(np.unique(indices)),
                 particle_sha256=hashlib.sha256(z1.numpy().tobytes()).hexdigest(),
                 weight_sha256=hashlib.sha256(weights.numpy().tobytes()).hexdigest(),
                 selection_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                 donor_identity_sha256=hashlib.sha256(donor_rows[indices].tobytes()).hexdigest(),
                 residual_provenance='Selected original donor expression and identity, unchanged past head coefficients',
                 growth_scope='Normalized conditional weights; absolute growth unidentifiable',
                 guard_reference='Selected original donor population', forecast_midpoint_max_step=.0625)
    return pred, indices, audit
