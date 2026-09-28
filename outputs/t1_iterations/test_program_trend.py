import numpy as np
from program_trend import ProgramTrend


def test_program_basis_and_forecast_exclude_future():
    rng=np.random.default_rng(80);stages=np.repeat([7.5,7.75,8.,9.],80)
    labels=np.tile(np.repeat(['A','B'],40),4)
    x=rng.uniform(.2,.8,(320,6)).astype(np.float32)
    for i in range(4):
        x[i*80:(i+1)*80,:2]+=.12*i
        x[i*80:(i+1)*80,2:4]*=1-.1*i
    x[::9,1]=0
    donors=np.column_stack([x[160:240],np.full(80,.4)]).astype(np.float32)
    panel=[f'G{i}' for i in range(6)]+['protected'];symbols=panel[:6]
    args=(8.,donors,labels[160:240],panel,symbols,np.arange(6))
    model=ProgramTrend(x,stages,labels,*args)
    changed=x.copy();changed[stages>8.]=99;altered=labels.copy();altered[stages>8.]='X'
    other=ProgramTrend(changed,stages,altered,*args)
    np.testing.assert_array_equal(model.basis,other.basis)
    original=model.positive_slope.copy()
    pred,_,audit=model.predict_program(9.,2)
    np.testing.assert_array_equal(pred,other.predict_program(9.,2)[0])
    np.testing.assert_array_equal(model.positive_slope,original)
    assert np.any(pred!=donors)
    np.testing.assert_array_equal(pred[:,6],donors[:,6])
    np.testing.assert_array_equal(pred==0,donors==0)
    np.testing.assert_allclose(np.expm1(pred[:,:6]).sum(1),np.expm1(donors[:,:6]).sum(1),rtol=1e-6)
    assert audit['covariance_change_vs_reference']<=.4
