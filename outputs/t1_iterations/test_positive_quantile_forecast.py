import numpy as np
from positive_quantile_forecast import PositiveQuantileForecast


def test_future_mutation_and_protected_zeros_mass():
    rng=np.random.default_rng(23);stages=np.repeat([7.5,7.75,8.,9.],40)
    x=rng.uniform(.3,1.2,(160,3)).astype(np.float32)
    x[:40]*=.7;x[40:80]*=.9;x[::7,0]=0
    donors=np.column_stack([x[80:120,:2],rng.uniform(.1,.6,40)]).astype(np.float32)
    donors[::6,1]=0
    args=(stages,8.,donors,['A','B','protected'],['A','B','C'],[0,1])
    model=PositiveQuantileForecast(x,*args,min_positive=20)
    changed=x.copy();changed[stages>8.]=100
    other=PositiveQuantileForecast(changed,*args,min_positive=20)
    np.testing.assert_array_equal(model.slope,other.slope)
    pred,rows,audit=model.predict(9.)
    assert np.any(pred!=donors)
    np.testing.assert_array_equal(pred[:,2],donors[:,2])
    np.testing.assert_array_equal(pred==0,donors==0)
    np.testing.assert_allclose(np.expm1(pred[:,:2]).sum(1),np.expm1(donors[:,:2]).sum(1),rtol=1e-6)
    assert audit['covariance_change_vs_reference']<=.4
