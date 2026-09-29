import numpy as np
import torch
from feature_panel_forecast import FeaturePanelForecast
from cnf_density_flow import DensityFlowNet


def test_feature_guard_hurdle_modes_remain_past_only_and_reproducible():
    rng=np.random.default_rng(481);stages=np.repeat([7.5,7.75,8.,9.],32)
    x=rng.uniform(.2,.8,(128,8)).astype(np.float32);x[rng.random(x.shape)<.3]=0
    donors=np.column_stack([x[64:96],np.full(32,.3)]).astype(np.float32)
    panel=[str(i) for i in range(8)]+['protected']
    torch.manual_seed(481);net=DensityFlowNet(np.eye(8)[:3],np.zeros(8),8.,7.25)
    with torch.no_grad():net.field[-1].bias.fill_(.1)
    args=(stages,8.,donors,panel,panel[:8],net,np.zeros(8,dtype=np.float32),np.ones(8,dtype=np.float32),np.arange(8),np.array([0,1]))
    model=FeaturePanelForecast(x,*args);changed=x.copy();changed[stages>8.]=99;other=FeaturePanelForecast(changed,*args)
    np.testing.assert_array_equal(model.detection,other.detection);np.testing.assert_array_equal(model.positive_coef,other.positive_coef)
    model.detection[:]=1.;other.detection[:]=1.
    for mode in ['detection','joint']:
        for sampling in ['independent','systematic']:
            prediction,_,audit=model.predict(9.,mode,1.,sampling=sampling)
            np.testing.assert_array_equal(prediction,other.predict(9.,mode,1.,sampling=sampling)[0])
            np.testing.assert_array_equal(prediction[:,-1],donors[:,-1])
            np.testing.assert_allclose(np.expm1(prediction[:,:8]).sum(1),np.expm1(donors[:,:8]).sum(1),rtol=1e-6)
            assert np.isfinite(prediction).all() and (prediction>=0).all()
            assert audit['covariance_change_vs_reference']<=.4
            assert audit['newly_detected_entries']>0
