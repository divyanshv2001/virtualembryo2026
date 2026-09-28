import numpy as np
from empirical_bayes_trend import shrink_normal_effects,EmpiricalBayesTrend


def test_normal_mixture_preserves_strong_sign_and_rejects_uncertainty():
    effects=np.r_[np.zeros(200),np.ones(100),-np.ones(100),.1]
    errors=np.r_[np.full(400,.02),1.]
    mean,risk,audit=shrink_normal_effects(effects,errors)
    assert audit['converged'] and audit['penalized_objective_monotone']
    assert mean[201]>.9 and mean[301]<-.9 and risk[201]<.01
    assert abs(mean[-1])<.1 and risk[-1]>.25
    np.testing.assert_allclose(mean[:200],0)


def test_bayes_forecast_excludes_future_data_and_preserves_protected_genes():
    rng=np.random.default_rng(991);stages=np.repeat([7.5,7.75,8.,9.],60);labels=np.repeat('A',240)
    x=rng.uniform(.2,.7,(240,8)).astype(np.float32)
    x[:,0]+=np.repeat([0,.1,.2,.3],60);x[::9,2]=0
    donors=np.column_stack([x[120:180],np.full(60,.3)]).astype(np.float32)
    panel=[str(i) for i in range(8)]+['protected'];args=(8.,donors,labels[120:180],panel,panel[:8],np.arange(8))
    model=EmpiricalBayesTrend(x,stages,labels,*args)
    changed=x.copy();changed[stages>8]=99
    other=EmpiricalBayesTrend(changed,stages,labels,*args)
    np.testing.assert_array_equal(model.positive_slope,other.positive_slope)
    pred,_,audit=model.predict_bayes(9.)
    np.testing.assert_array_equal(pred[:,8],donors[:,8]);np.testing.assert_array_equal(pred==0,donors==0)
    np.testing.assert_allclose(np.expm1(pred[:,:8]).sum(1),np.expm1(donors[:,:8]).sum(1),rtol=1e-6)
    assert audit['covariance_change_vs_reference']<=.4
