import numpy as np
import torch
from cnf_density_flow import DensityFlowNet,inverse_density
from cnf_autonomous_flow import AutonomousDensityFlowNet


def test_autonomy_nonzero_field_and_finite_density_gradients():
    torch.manual_seed(41);net=AutonomousDensityFlowNet(np.eye(2),np.zeros(2),8.5,7.25)
    with torch.no_grad():net.field[-1].weight.normal_(0,.03)
    z=torch.randn(7,2)
    first=net.velocity(.1,z)
    assert first.abs().sum()>0
    torch.testing.assert_close(first,net.velocity(9.,z),rtol=0,atol=0)
    with torch.no_grad():
        direct=net.trajectory(z,torch.tensor([0.,1.]))[-1]
        segmented=net.trajectory(z,torch.tensor([0.,.5,1.]))[-1]
    torch.testing.assert_close(direct,segmented,rtol=0,atol=0)
    nll,energy=inverse_density(net.velocity,z.clone(),.5,noise=torch.ones_like(z))
    (nll.mean()+.1*energy.mean()).backward()
    assert torch.isfinite(nll).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.field.parameters())
    torch.testing.assert_close(net.field[0].weight.grad[:,-1],torch.zeros(64),rtol=0,atol=0)


def test_autonomous_keeps_explicit_initialization_lineage():
    torch.manual_seed(41);first=DensityFlowNet(np.eye(2),np.zeros(2),8.5,7.25)
    torch.manual_seed(41);second=AutonomousDensityFlowNet(np.eye(2),np.zeros(2),8.5,7.25)
    for key,value in first.state_dict().items():torch.testing.assert_close(value,second.state_dict()[key],rtol=0,atol=0)
