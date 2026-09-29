import numpy as np
import torch
from cnf_density_flow import DensityFlowNet
from latent_dispersion import LatentDispersion


def test_zero_exact_seed_replay_antithetic_mean_and_time_scaling():
    torch.manual_seed(4);net=DensityFlowNet(np.eye(3),np.zeros(3),8.5,7.25)
    with torch.no_grad():net.field[-1].bias.fill_(.1)
    z=torch.randn(101,3);times=torch.tensor([0.,1.,4.])
    base=net.trajectory(z,times);model=LatentDispersion(net)
    assert torch.equal(base,model.trajectory(z,times))
    model.configure(.05);first=model.trajectory(z,times)
    assert torch.equal(first,model.trajectory(z,times))
    torch.testing.assert_close(first[0],base[0],rtol=0,atol=0)
    torch.testing.assert_close((first[1]-base[1]).mean(0),torch.zeros(3),rtol=0,atol=1e-7)
    torch.testing.assert_close(first[2]-base[2],2*(first[1]-base[1]),rtol=1e-4,atol=3e-7)
    torch.testing.assert_close(model.encode(z)[0],net.encode(z)[0],rtol=0,atol=0)
