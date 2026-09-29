import numpy as np
import pytest
from strength_transport_forecast import StrengthTransportForecast
from conditional_positive_head import FullConditionalPositiveForecast
from transport_latent_flow import AffineTransportNet


def test_strength_control_matches_old_and_preserves_fitted_heads_and_guards():
    rng=np.random.default_rng(435);stages=np.repeat([7.5,7.75,8.,9.],32)
    x=rng.uniform(.2,.8,(128,8)).astype(np.float32);x[rng.random(x.shape)<.3]=0
    donors=np.column_stack([x[64:96],np.full(32,.3)]).astype(np.float32)
    panel=[str(i) for i in range(8)]+['protected']
    coefficient=np.zeros((4,3));coefficient[-1]=.5
    net=AffineTransportNet(np.eye(8)[:3],np.zeros(8),coefficient)
    args=(stages,8.,donors,panel,panel[:8],net,np.zeros(8,dtype=np.float32),np.ones(8,dtype=np.float32),np.arange(8))
    model=StrengthTransportForecast(x,*args);baseline=FullConditionalPositiveForecast(x,*args)
    fitted=model.positive_coef.copy()
    for strength in [.5,1.]:
        np.testing.assert_array_equal(model.predict(9.,'abundance',strength)[0],baseline.predict(9.,'abundance',strength)[0])
    for strength in [.25,.5,1.,2.]:
        pred,_,audit=model.predict(9.,'abundance',strength)
        np.testing.assert_array_equal(model.positive_coef,fitted)
        np.testing.assert_array_equal(pred[:,-1],donors[:,-1])
        np.testing.assert_array_equal(pred==0,donors==0)
        np.testing.assert_allclose(np.expm1(pred[:,:8]).sum(1),np.expm1(donors[:,:8]).sum(1),rtol=1e-6)
        assert audit['covariance_change_vs_reference']<=.4
        assert audit['strength']==strength
    with pytest.raises(ValueError):model.predict(9.,'abundance',3.)
    with pytest.raises(ValueError):model.predict(8.,'abundance',2.)
    np.testing.assert_array_equal(model.positive_coef,fitted)
