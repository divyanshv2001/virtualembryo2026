import copy
import numpy as np
import torch
from cnf_manifold_flow import DensityFlowNet,train_manifold_density
from cnf_noise_flow import train_noise_density


def test_zero_noise_exact_control_and_noisy_checkpoint_resume(tmp_path):
    torch.manual_seed(448);net=DensityFlowNet(np.eye(2),np.zeros(2),8.,7.5)
    values=np.random.default_rng(448).normal(size=(48,2)).astype(np.float32);stages=np.repeat([7.75,8.],24)
    base=copy.deepcopy(net);zero=copy.deepcopy(net)
    train_manifold_density(base,values,stages,.1,tmp_path/'base.pt',lambda *a,**k:None,steps=2,density_weight=10.)
    train_noise_density(zero,values,stages,.1,tmp_path/'zero.pt',lambda *a,**k:None,steps=2,density_weight=10.)
    for key,value in base.state_dict().items():torch.testing.assert_close(value,zero.state_dict()[key],rtol=0,atol=0)
    noisy=copy.deepcopy(net);path=tmp_path/'noisy.pt'
    train_noise_density(noisy,values,stages,.1,path,lambda *a,**k:None,steps=2,density_weight=10.,observation_noise=.1)
    restored=copy.deepcopy(net)
    train_noise_density(restored,values,stages,.1,path,lambda *a,**k:None,resume=True,steps=2,density_weight=10.,observation_noise=.1)
    for key,value in noisy.state_dict().items():torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
    assert all(torch.isfinite(p).all() for p in noisy.parameters())
