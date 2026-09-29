import numpy as np
from growth_proxy_forecast import GrowthProxyComposition


def test_neutral_identity_and_whole_cell_weighted_prediction():
    rng=np.random.default_rng(9);donors=rng.uniform(.1,.7,(64,8)).astype(np.float32)
    model=GrowthProxyComposition(donors,np.r_[np.zeros(32),np.ones(32)],np.zeros(64),np.arange(8),8.)
    neutral,indices,_=model.predict(9.,'net',0.)
    np.testing.assert_array_equal(indices,np.arange(64));np.testing.assert_array_equal(neutral,donors)
    pred,indices,audit=model.predict(9.,'net',.5)
    np.testing.assert_array_equal(pred,donors[indices]);np.testing.assert_array_equal(pred,model.predict(9.,'net',.5)[0])
    assert (indices>=32).sum()>32 and len(pred)==64
    assert audit['covariance_change_vs_reference']<=.4
