import numpy as np
from neural_ode_forecast import NeuralODEForecast
from neural_hurdle_forecast import NeuralHurdleForecast


def test_hurdle_heads_are_past_only_and_changes_are_reproducible():
    rng=np.random.default_rng(191);stages=np.repeat([7.5,7.75,8.,9.],32)
    x=rng.uniform(.2,.8,(128,8)).astype(np.float32);x[rng.random(x.shape)<.35]=0
    donors=np.column_stack([x[64:96],np.full(32,.3)]).astype(np.float32)
    panel=[str(i) for i in range(8)]+['protected']
    neural=NeuralODEForecast(x,stages,8.,donors,panel,panel[:8],np.arange(8),beta=0.,pretrain_steps=2,joint_steps=2,batch=8,truth_batch=16)
    args=(stages,8.,donors,panel,panel[:8],neural.net,neural.center,neural.scale,neural.features)
    model=NeuralHurdleForecast(x,*args);changed=x.copy();changed[stages>8]=99
    other=NeuralHurdleForecast(changed,*args)
    np.testing.assert_array_equal(model.detection,other.detection)
    np.testing.assert_array_equal(model.positive_coef,other.positive_coef)
    # Deliberately force a supported probability change so the transition branch is exercised.
    model.detection[:]=1.;other.detection[:]=1.
    pred,indices,audit=model.predict(9.,'joint');repeat=other.predict(9.,'joint')[0]
    np.testing.assert_array_equal(pred,repeat);np.testing.assert_array_equal(indices,np.arange(32))
    np.testing.assert_array_equal(pred[:,8],donors[:,8])
    np.testing.assert_allclose(np.expm1(pred[:,:8]).sum(1),np.expm1(donors[:,:8]).sum(1),rtol=1e-6)
    assert np.isfinite(pred).all() and (pred>=0).all()
    assert audit['newly_detected_entries']+audit['removed_detection_entries']>0
    assert audit['covariance_change_vs_reference']<=.4
    abundance=model.predict(9.,'abundance')[0]
    np.testing.assert_array_equal(abundance==0,donors==0)
