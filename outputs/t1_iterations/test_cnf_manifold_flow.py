import copy
import numpy as np
import torch
from cnf_density_flow import DensityFlowNet,train_density
from cnf_manifold_flow import train_manifold_density,manifold_penalty,rk4_position


def test_manifold_hinge_and_midpoint_gradient_and_zero_regularizer_equivalence(tmp_path):
    reference=torch.tensor([[0.,0.],[.05,0.],[0.,.05],[.05,.05],[-.05,0.]])
    near=torch.tensor([[0.,0.]],requires_grad=True);torch.testing.assert_close(manifold_penalty(near,reference),torch.tensor(0.))
    far=torch.tensor([[2.,2.]],requires_grad=True);loss=manifold_penalty(far,reference);loss.backward();assert far.grad.abs().sum()>0
    z=torch.tensor([[1.,2.]]);flow=lambda t,v:v*.1
    midpoint=rk4_position(flow,z,1.,.875)
    torch.testing.assert_close(midpoint,z*float(np.exp(-.0125)))
    torch.manual_seed(445);base=DensityFlowNet(np.eye(2),np.zeros(2),8.,7.5);other=copy.deepcopy(base)
    values=np.random.default_rng(445).normal(size=(48,2)).astype(np.float32);stages=np.repeat([7.75,8.],24)
    train_density(base,values,stages,.1,tmp_path/'base.pt',lambda *a,**k:None,steps=2)
    train_manifold_density(other,values,stages,.1,tmp_path/'other.pt',lambda *a,**k:None,steps=2,density_weight=0.)
    for key,value in base.state_dict().items():torch.testing.assert_close(value,other.state_dict()[key],rtol=0,atol=0)


def test_positive_manifold_training_has_finite_gradients_at_reference_cells(tmp_path):
    torch.manual_seed(446);net=DensityFlowNet(np.eye(2),np.zeros(2),8.,7.5)
    values=np.random.default_rng(446).normal(size=(48,2)).astype(np.float32);stages=np.repeat([7.75,8.],24)
    history=train_manifold_density(net,values,stages,.1,tmp_path/'positive.pt',lambda *a,**k:None,steps=2,density_weight=10.)
    assert np.isfinite(history[-1]['loss'])
    assert all(torch.isfinite(v).all() for v in net.parameters())
