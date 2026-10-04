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
