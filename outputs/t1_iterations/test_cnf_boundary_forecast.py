import torch
from cnf_boundary_forecast import BoundaryFlow


class Analytic:
    cutoff, origin = 8.5, 7.25
    def __init__(self): self.calls = []
    def velocity(self, t, z):
        self.calls.append(t)
        return t * z
    def encode(self, x): return x, torch.zeros_like(x)
    def trajectory(self, z, times, step=.125): return torch.stack([z, z+7])


def test_timeclamp_matches_autonomous_analytic_solution():
    net = Analytic()
    z = torch.tensor([[1., 2.]])
    out = BoundaryFlow(net, 'timeclamp').trajectory(z, torch.tensor([0., .5, 1.]), step=.03125)
    torch.testing.assert_close(out[-1], z * torch.exp(torch.tensor(1.25)), rtol=1e-5, atol=1e-5)
    assert set(net.calls) == {1.25}


def test_ballistic_and_original_have_declared_behavior():
    net = Analytic(); z = torch.tensor([[1., 2.]])
    out = BoundaryFlow(net, 'ballistic').trajectory(z, torch.tensor([0., .5, 1.]))
    torch.testing.assert_close(out[-1], z * 2.25)
    assert len(net.calls) == 1
    torch.testing.assert_close(BoundaryFlow(net, 'original').trajectory(z, torch.tensor([0., 1.])), net.trajectory(z, torch.tensor([0., 1.])))
