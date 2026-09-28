import numpy as np
from copula_trend import couple_positive_margins


def test_rank_coupling_keeps_exact_margins_and_zero_positions():
    original=np.array([[0.,1.],[3.,0.],[2.,4.],[1.,2.]])
    scores=np.array([[8.,9.],[1.,9.],[2.,2.],[3.,1.]])
    pred=couple_positive_margins(original,scores)
    np.testing.assert_array_equal(pred==0,original==0)
    np.testing.assert_array_equal(np.sort(pred,axis=0),np.sort(original,axis=0))
    np.testing.assert_array_equal(pred[:,0],[0.,1.,2.,3.])
    assert np.isfinite(pred).all()
