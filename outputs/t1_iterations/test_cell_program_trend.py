import numpy as np
from cell_program_trend import CellProgramTrend


def test_cell_programs_are_past_only_and_conserve_protected_mass():
    rng=np.random.default_rng(95);stages=np.repeat([7.5,7.75,8.,9.],80)
    labels=np.tile(np.repeat(['A','B'],40),4)
    x=rng.uniform(.2,.6,(320,8)).astype(np.float32)
    for i in range(4):
        x[i*80:(i+1)*80,:2]+=.14*i
        x[i*80:(i+1)*80,2:4]*=1-.12*i
    x[::8,6]=0
    donors=np.column_stack([x[160:240],np.full(80,.3)]).astype(np.float32)
    panel=[f'G{i}' for i in range(8)]+['protected'];symbols=panel[:8]
    args=(8.,donors,labels[160:240],panel,symbols,np.arange(8))
    model=CellProgramTrend(x,stages,labels,*args)
    changed=x.copy();changed[stages>8.]=99;altered=labels.copy();altered[stages>8.]='X'
    other=CellProgramTrend(changed,stages,altered,*args)
    np.testing.assert_array_equal(model.consensus,other.consensus)
    np.testing.assert_array_equal(model.decoder,other.decoder)
    pred,indices,audit=model.predict_cell_program(9.)
    np.testing.assert_array_equal(pred,other.predict_cell_program(9.)[0])
    assert np.any(pred!=donors)
    np.testing.assert_array_equal(indices,np.arange(80))
    np.testing.assert_array_equal(pred[:,8],donors[:,8])
    np.testing.assert_array_equal(pred==0,donors==0)
    np.testing.assert_allclose(np.expm1(pred[:,:8]).sum(1),np.expm1(donors[:,:8]).sum(1),rtol=1e-6)
    assert audit['covariance_change_vs_reference']<=.4
