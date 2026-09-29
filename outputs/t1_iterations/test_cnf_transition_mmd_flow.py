import copy
import numpy as np
import torch
from cnf_density_flow import DensityFlowNet
from cnf_manifold_flow import train_manifold_density
from cnf_transition_mmd_flow import kernel_mmd,train_transition_mmd


def test_mmd_symmetry_equal_distributions_and_translation_gradient():
    torch.manual_seed(531);x=torch.randn(32,3);shift=torch.tensor(1.,requires_grad=True)
    torch.testing.assert_close(kernel_mmd(x,x,2.),torch.tensor(0.))
    loss=kernel_mmd(x,x+shift,2.);loss.backward();assert loss>0 and shift.grad>0
    torch.testing.assert_close(loss,kernel_mmd(x+1.,x,2.))


def test_zero_mmd_weight_replays_unmodified_training_continuation(tmp_path):
    torch.manual_seed(532);base=DensityFlowNet(np.eye(2),np.zeros(2),8.,7.25)
    z=np.random.default_rng(532).normal(size=(48,2)).astype(np.float32);stages=np.repeat([7.5,7.75,8.],16)
    train_manifold_density(base,z,stages,.1,tmp_path/'initial.pt',lambda *a,**k:None,steps=2,density_weight=10.)
    adapted=copy.deepcopy(base)
    train_transition_mmd(adapted,z,stages,tmp_path/'initial.pt',tmp_path/'adapted.pt',lambda *a,**k:None,weight=0.,updates=2)
    saved=torch.load(tmp_path/'initial.pt',weights_only=False);saved['steps']=4;torch.save(saved,tmp_path/'base.pt')
    train_manifold_density(base,z,stages,.1,tmp_path/'base.pt',lambda *a,**k:None,steps=4,density_weight=10.,resume=True)
    for key,value in base.state_dict().items():torch.testing.assert_close(value,adapted.state_dict()[key],rtol=0,atol=0)
