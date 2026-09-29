import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from logistic_detection_head import fit_logistic


def test_logistic_solver_matches_independent_optimizer_and_bounds_extremes():
    rng=np.random.default_rng(501);h=rng.normal(size=(300,2));y=(rng.random((300,2))<expit(h@np.array([[1.,-.5],[.2,.8]]))).astype(float)
    coef,audit=fit_logistic(h,y,1.)
    design=np.column_stack([h,np.ones(len(h))])
    for gene in range(2):
        def objective(c):
            v=design@c
            return np.mean(np.logaddexp(0,v)-y[:,gene]*v)+.5*(c[:-1]**2).sum()
        independent=minimize(objective,np.zeros(3),method='BFGS').x
        np.testing.assert_allclose(coef[:,gene],independent,atol=2e-4)
    assert audit['final_mean_objective']<audit['initial_mean_objective']
    assert np.isfinite(expit(np.array([-1e10,1e10]))).all()


def test_logistic_forecast_is_past_only_and_preserves_protected_genes():
    import torch
    from logistic_detection_head import LogisticDetectionForecast
    from cnf_density_flow import DensityFlowNet
    rng=np.random.default_rng(502);stages=np.repeat([7.5,7.75,8.,9.],32)
    x=rng.uniform(.2,.8,(128,8)).astype(np.float32);x[rng.random(x.shape)<.3]=0
    donors=np.column_stack([x[64:96],np.full(32,.3)]).astype(np.float32);panel=[str(i) for i in range(8)]+['protected']
    torch.manual_seed(502);net=DensityFlowNet(np.eye(8)[:3],np.zeros(8),8.,7.25)
    with torch.no_grad():net.field[-1].bias.fill_(.1)
    args=(stages,8.,donors,panel,panel[:8],net,np.zeros(8,dtype=np.float32),np.ones(8,dtype=np.float32),np.arange(8),np.array([0,1]))
    first=LogisticDetectionForecast(x,*args);changed=x.copy();changed[stages>8.]=99;second=LogisticDetectionForecast(changed,*args)
    np.testing.assert_array_equal(first.logistic_coef,second.logistic_coef)
    prediction,_,audit=first.predict(9.,'joint',1.,sampling='systematic')
    np.testing.assert_array_equal(prediction[:,-1],donors[:,-1]);assert np.isfinite(prediction).all()
    np.testing.assert_allclose(np.expm1(prediction[:,:8]).sum(1),np.expm1(donors[:,:8]).sum(1),rtol=1e-6)
    assert audit['covariance_change_vs_reference']<=.4
