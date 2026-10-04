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
