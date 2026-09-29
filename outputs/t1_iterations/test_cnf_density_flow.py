import math
import numpy as np
import torch
from cnf_density_flow import inverse_density,divergence,DensityFlowNet


def test_cnf_inverse_density_sign_energy_and_exact_trace():
    z=torch.tensor([[.2,.4],[-.3,.1]],requires_grad=True)
    rate=torch.tensor([.1,.2]);field=lambda t,v:v*rate
    nll,energy=inverse_density(field,z,.75,step=.025)
    initial=z.detach().numpy()*np.exp(-rate.numpy()*.75)
    expected=.5*(initial**2+math.log(2*math.pi)).sum(1)+float(rate.sum())*.75
    np.testing.assert_allclose(nll.detach().numpy(),expected,atol=1e-6)
    expected_energy=(z.detach().numpy()**2*rate.numpy()/2*(1-np.exp(-2*rate.numpy()*.75))).sum(1)
    np.testing.assert_allclose(energy.detach().numpy(),expected_energy,atol=1e-6)
    np.testing.assert_allclose(divergence(field(0,z),z,torch.ones_like(z)).detach().numpy(),np.full(2,.3),atol=1e-7)


def test_zero_density_network_is_identity_and_has_trainable_likelihood():
    torch.manual_seed(4);net=DensityFlowNet(np.eye(3),np.zeros(3),8.5,7.25)
    z=torch.randn(8,3);nll,energy=inverse_density(net.velocity,z.clone(),1.25,noise=torch.ones_like(z))
    expected=.5*(z.square()+math.log(2*math.pi)).sum(1)
    torch.testing.assert_close(nll,expected);torch.testing.assert_close(energy,torch.zeros(8))
    nll.mean().backward();assert net.field[-1].weight.grad.abs().sum()>0
    torch.testing.assert_close(net.trajectory(z,torch.tensor([0.,1.]))[-1],z)
