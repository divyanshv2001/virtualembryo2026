import numpy as np
from scipy.optimize import minimize
from poisson_positive_forecast import fit_positive_poisson,PoissonPositiveForecast
from feature_panel_forecast import FeaturePanelForecast
from transport_latent_flow import AffineTransportNet


def test_gamma_solver_matches_independent_loglink_objective():
    rng=np.random.default_rng(613);h=rng.normal(size=(200,2))
    y=rng.gamma(2,np.exp(.3+h@np.array([.4,-.2]))/2)[:,None]
    y[rng.random((200,1))<.2]=0
    coef,audit=fit_positive_poisson(h,y,family='gamma')
    a=np.column_stack([h,np.ones(len(h))]);mask=y[:,0]>0
    def loss(c):
        eta=a[mask]@c
        return np.mean(y[mask,0]*np.exp(-eta)+eta)+.5*np.sum(c[:-1]**2)
    reference=minimize(loss,np.zeros(3),method='BFGS').x
    np.testing.assert_allclose(coef[:,0],reference,atol=2e-4)
    assert audit['final_mean_objective']<=audit['initial_mean_objective']


def test_gamma_head_preserves_detection_and_excludes_future():
    rng=np.random.default_rng(614);stages=np.repeat([7.5,7.75,8.,9.],40)
    x=rng.uniform(.2,.8,(160,8)).astype(np.float32);x[rng.random(x.shape)<.3]=0
    donors=np.column_stack([x[80:120],np.full(40,.3)]).astype(np.float32)
    panel=[str(i) for i in range(8)]+['protected'];coefficient=np.zeros((4,3));coefficient[-1]=.1
    net=AffineTransportNet(np.eye(8)[:3],np.zeros(8),coefficient)
    args=(x,stages,8.,donors,panel,panel[:8],net,np.zeros(8,dtype=np.float32),np.ones(8,dtype=np.float32),np.arange(8),np.arange(8))
    model=PoissonPositiveForecast(*args,positive_family='gamma');control=FeaturePanelForecast(*args)
    np.testing.assert_array_equal(model.detection,control.detection)
    altered=x.copy();altered[stages>8.]=999
    other=PoissonPositiveForecast(altered,*args[1:],positive_family='gamma')
    pred=model.predict(9.,'joint',1.,sampling='systematic')[0]
    np.testing.assert_array_equal(pred,other.predict(9.,'joint',1.,sampling='systematic')[0])
    np.testing.assert_array_equal(pred[:,-1],donors[:,-1])
    np.testing.assert_allclose(np.expm1(pred[:,:8]).sum(1),np.expm1(donors[:,:8]).sum(1),rtol=1e-6)
