import torch
from latent_population_scale import LatentPopulationScale


class FixedFlow:
    cutoff = 8.5
    origin = 7.25

    def encode(self, x):
        return x, None

    def trajectory(self, z, times):
        return torch.stack([z + t * .1 for t in times])


def test_exact_identity_mean_covariance_and_initial_condition():
    z = torch.tensor([[1., 2.], [3., 5.], [-2., 1.]], dtype=torch.float64)
    times = torch.tensor([0., 1., 2.], dtype=torch.float64)
    flow = FixedFlow()
    base = flow.trajectory(z, times)
    model = LatentPopulationScale(flow)
    assert torch.equal(base, model.trajectory(z, times))
    for factor in [.75, 1.25]:
        model.configure(factor)
        result = model.trajectory(z, times)
        assert torch.equal(base[0], result[0])
        for i in [1, 2]:
            torch.testing.assert_close(result[i].mean(0), base[i].mean(0))
            torch.testing.assert_close(torch.cov(result[i].T), torch.cov(base[i].T) * factor ** (2 * i))
        assert torch.equal(model.encode(z)[0], z)
