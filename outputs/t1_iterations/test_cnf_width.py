import numpy as np
import torch
from cnf_density_flow import DensityFlowNet,inverse_density


def test_width64_default_lineage_and_wide_density_gradients():
    torch.manual_seed(812);default=DensityFlowNet(np.eye(4),np.zeros(4),8.,7.25)
    torch.manual_seed(812);explicit=DensityFlowNet(np.eye(4),np.zeros(4),8.,7.25,width=64)
    for key,value in default.state_dict().items():torch.testing.assert_close(value,explicit.state_dict()[key],rtol=0,atol=0)
    for width in [128,256]:
        torch.manual_seed(812);net=DensityFlowNet(np.eye(4),np.zeros(4),8.,7.25,width=width)
        z=torch.randn(8,4)
        torch.testing.assert_close(net.velocity(.5,z),torch.zeros_like(z),rtol=0,atol=0)
        nll,energy=inverse_density(net.velocity,z,.5,noise=torch.ones_like(z))
        loss=nll.mean()+.1*energy.mean();loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.field.parameters())
        assert net.field[-1].weight.shape==(4,width)
