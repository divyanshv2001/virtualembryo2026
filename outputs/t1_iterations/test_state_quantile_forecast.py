import numpy as np
from state_quantile_forecast import StateQuantileForecast


def test_conditional_forecast_is_past_only_and_preserves_cells():
    rng=np.random.default_rng(52);stages=np.repeat([7.5,7.75,8.,9.],160)
    x=np.zeros((640,8),np.float32)
    for t in range(4):
        for k in range(4):
            rows=slice(t*160+k*40,t*160+(k+1)*40)
            x[rows,:4]=.1+rng.uniform(0,.02,(40,4));x[rows,k]+=3
            x[rows,4:]=.3+.03*t+rng.uniform(0,.15,(40,4))
    x[::11,5]=0
    donors=np.column_stack([x[320:480],np.full(160,.27)]).astype(np.float32)
    panel=[f'G{i}' for i in range(8)]+['protected'];symbols=panel[:8]
    args=(stages,8.,donors,panel,symbols,np.arange(8))
    model=StateQuantileForecast(x,*args)
    future_changed=x.copy();future_changed[stages>8.]=50
    other=StateQuantileForecast(future_changed,*args)
    pred,indices,audit=model.predict(9.)
    other_pred,_,_=other.predict(9.)
    assert model.models and np.any(pred!=donors)
    np.testing.assert_array_equal(pred,other_pred)
    np.testing.assert_array_equal(indices,np.arange(160))
    np.testing.assert_array_equal(pred[:,8],donors[:,8])
    np.testing.assert_array_equal(pred==0,donors==0)
    np.testing.assert_allclose(np.expm1(pred[:,:8]).sum(1),np.expm1(donors[:,:8]).sum(1),rtol=1e-6)
    assert audit['covariance_change_vs_reference']<=.4
