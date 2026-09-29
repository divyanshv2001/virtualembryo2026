import numpy as np
from feature_panel_forecast import FeaturePanelForecast
from ridge_conditional_head import RidgeConditionalPositiveForecast
from transport_latent_flow import AffineTransportNet
from robust_population import covariance_change


def test_fixed_guard_panel_control_and_expanded_encoding():
    rng=np.random.default_rng(451);stages=np.repeat([7.5,7.75,8.,9.],32)
    x=rng.uniform(.2,.8,(128,8)).astype(np.float32);x[rng.random(x.shape)<.3]=0
    donors=np.column_stack([x[64:96],np.full(32,.3)]).astype(np.float32);panel=[str(i) for i in range(8)]+['protected']
    coefficient=np.zeros((4,3));coefficient[-1]=.1;net=AffineTransportNet(np.eye(8)[:3],np.zeros(8),coefficient)
    args=(stages,8.,donors,panel,panel[:8],net,np.zeros(8,dtype=np.float32),np.ones(8,dtype=np.float32),np.arange(8))
    baseline=RidgeConditionalPositiveForecast(x,*args);control=FeaturePanelForecast(x,*args,np.arange(8))
    np.testing.assert_array_equal(control.predict(9.,'abundance',1.)[0],baseline.predict(9.,'abundance',1.)[0])
    expanded=FeaturePanelForecast(x,*args,np.array([0,1]))
    pred,_,audit=expanded.predict(9.,'abundance',1.)
    assert abs(audit['covariance_change_vs_reference']-covariance_change(donors[:,:2],pred[:,:2]))<1e-12
    np.testing.assert_array_equal(pred[:,-1],donors[:,-1]);np.testing.assert_array_equal(pred==0,donors==0)
    np.testing.assert_allclose(np.expm1(pred[:,:8]).sum(1),np.expm1(donors[:,:8]).sum(1),rtol=1e-6)
