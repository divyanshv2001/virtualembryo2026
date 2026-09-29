import numpy as np
import pytest
import torch
from cnf_density_flow import DensityFlowNet
from cnf_manifold_flow import train_manifold_density


def test_batch_default_exact_and_large_batch_checkpoint(tmp_path):
    z=np.random.default_rng(12).normal(size=(16,2)).astype(np.float32)
    stages=np.repeat([7.5,7.75],8)
    def make():
        torch.manual_seed(17)
        return DensityFlowNet(np.eye(2),np.zeros(2),7.75,7.25)
    default,explicit=make(),make()
    for net,path,kwargs in [(default,tmp_path/'a.pt',{}),(explicit,tmp_path/'b.pt',{'batch_size':64})]:
        train_manifold_density(net,z,stages,.1,path,lambda *a,**kw:None,steps=2,density_weight=10.,**kwargs)
    for key,value in default.state_dict().items():
        torch.testing.assert_close(value,explicit.state_dict()[key],rtol=0,atol=0)
    large=make();path=tmp_path/'large.pt'
    records=train_manifold_density(large,z,stages,.1,path,lambda *a,**kw:None,steps=2,density_weight=10.,batch_size=256)
    assert np.isfinite(records[-1]['loss'])
    assert torch.load(path,weights_only=False)['batch_size']==256
    with pytest.raises(ValueError,match='Checkpoint configuration'):
        train_manifold_density(make(),z,stages,.1,path,lambda *a,**kw:None,resume=True,steps=2,density_weight=10.,batch_size=64)
    with pytest.raises(ValueError,match='Undeclared batch'):
        train_manifold_density(make(),z,stages,.1,path,lambda *a,**kw:None,steps=2,batch_size=128)
