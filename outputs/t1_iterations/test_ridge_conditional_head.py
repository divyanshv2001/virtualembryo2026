import numpy as np
from ridge_conditional_head import RidgeConditionalPositiveForecast
from conditional_positive_head import FullConditionalPositiveForecast, conditional_ridge
from transport_latent_flow import AffineTransportNet


def test_ridge_control_matches_and_future_invariance_and_regularization_effect():
    rng=np.random.default_rng(437);stages=np.repeat([7.5,7.75,8.,9.],32)
    x=rng.uniform(.2,.8,(128,8)).astype(np.float32);x[rng.random(x.shape)<.3]=0
    donors=np.column_stack([x[64:96],np.full(32,.3)]).astype(np.float32)
    panel=[str(i) for i in range(8)]+['protected']
    coefficient=np.zeros((4,3));coefficient[-1]=.1
    net=AffineTransportNet(np.eye(8)[:3],np.zeros(8),coefficient)
    args=(stages,8.,donors,panel,panel[:8],net,np.zeros(8,dtype=np.float32),np.ones(8,dtype=np.float32),np.arange(8))
    control=RidgeConditionalPositiveForecast(x,*args,ridge=1.)
    baseline=FullConditionalPositiveForecast(x,*args)
    np.testing.assert_array_equal(control.positive_coef,baseline.positive_coef)
    np.testing.assert_array_equal(control.predict(9.,'abundance',.5)[0],baseline.predict(9.,'abundance',.5)[0])
    changed=x.copy();changed[stages>8]=99
    low=RidgeConditionalPositiveForecast(x,*args,ridge=.1)
    other=RidgeConditionalPositiveForecast(changed,*args,ridge=.1)
    np.testing.assert_array_equal(low.positive_coef,other.positive_coef)
    assert np.linalg.norm(low.positive_coef)>np.linalg.norm(control.positive_coef)
    pred,_,audit=low.predict(9.,'abundance',.5)
    np.testing.assert_array_equal(pred,other.predict(9.,'abundance',.5)[0])
    np.testing.assert_array_equal(pred[:,-1],donors[:,-1])
    np.testing.assert_array_equal(pred==0,donors==0)
    np.testing.assert_allclose(np.expm1(pred[:,:8]).sum(1),np.expm1(donors[:,:8]).sum(1),rtol=1e-6)
    assert audit['covariance_change_vs_reference']<=.4
