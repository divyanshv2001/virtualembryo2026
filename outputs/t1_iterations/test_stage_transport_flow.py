import numpy as np
import torch
from stage_transport_flow import StageTransportFlow,StageTransportNet
from transport_latent_flow import TemporalTransportFlow


def test_stage_time_integration_uses_elapsed_time_and_matches_linear_acceleration():
    # No state dependence: velocity=.1*(t/.25)+.2, hence integral0..1=.4.
    net=StageTransportNet(np.ones((1,1)),np.zeros(1),np.array([[0.],[.1],[.2]]))
    z=torch.zeros((3,1));one=net.trajectory(z,torch.tensor([0.,1.]))[-1]
    split=net.trajectory(z,torch.tensor([0.,.5,1.]))[-1]
    torch.testing.assert_close(one,torch.full_like(one,.4));torch.testing.assert_close(one,split)


def test_stage_flow_fit_is_past_only_and_time_disabled_matches_baseline():
    rng=np.random.default_rng(191);x=rng.uniform(.1,1.,(128,8)).astype(np.float32)
    stages=np.repeat([7.5,7.75,8.,9.],32);rows=np.flatnonzero(stages<=8.)
    args=(stages,8.,[str(i) for i in range(8)],[str(i) for i in range(8)],np.arange(8),rows,np.zeros(len(rows)),'neutral')
    baseline=TemporalTransportFlow(x,*args,cell_budget=16)
    disabled=StageTransportFlow(x,*args,cell_budget=16,time_ridge=None)
    np.testing.assert_array_equal(baseline.net.coefficient.numpy(),disabled.net.coefficient.numpy())
    model=StageTransportFlow(x,*args,cell_budget=16,time_ridge=1.)
    changed=x.copy();changed[stages>8]=99
    other=StageTransportFlow(changed,*args,cell_budget=16,time_ridge=1.)
    np.testing.assert_array_equal(model.net.coefficient.numpy(),other.net.coefficient.numpy())
    assert model.audit['fit_max_stage']==8.
