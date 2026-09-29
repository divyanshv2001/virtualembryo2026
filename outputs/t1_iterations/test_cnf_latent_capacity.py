import numpy as np
import pytest
import torch
from cnf_density_flow import DensityFlowNet
from feature_panel_forecast import FeaturePanelForecast


@pytest.mark.parametrize('dimension',[16,24])
def test_full_conditional_decoder_with_larger_latent_rank(dimension):
    rng=np.random.default_rng(462);stages=np.repeat([7.5,7.75,8.,9.],32)
    x=rng.uniform(.2,.8,(128,28)).astype(np.float32);x[rng.random(x.shape)<.2]=0
    donors=np.column_stack([x[64:96],np.full(32,.3)]).astype(np.float32)
    panel=[str(i) for i in range(28)]+['protected']
    torch.manual_seed(462);net=DensityFlowNet(np.eye(28,dtype=np.float32)[:dimension],np.zeros(28),8.,7.25)
    with torch.no_grad():net.field[-1].bias.fill_(.05)
    forecast=FeaturePanelForecast(x,stages,8.,donors,panel,panel[:28],net,np.zeros(28,dtype=np.float32),np.ones(28,dtype=np.float32),np.arange(28),np.arange(5))
    assert forecast.positive_coef.shape==(dimension,28)
    prediction,_,audit=forecast.predict(9.,'abundance',1.)
    assert np.isfinite(prediction).all() and (prediction>=0).all()
    np.testing.assert_array_equal(prediction[:,-1],donors[:,-1])
    np.testing.assert_array_equal(prediction==0,donors==0)
    np.testing.assert_allclose(np.expm1(prediction[:,:28]).sum(1),np.expm1(donors[:,:28]).sum(1),rtol=1e-6)
    assert audit['covariance_guard_features']==5
