import numpy as np
from recent_positive_forecast import RecentPositiveForecast
from feature_panel_forecast import FeaturePanelForecast
from transport_latent_flow import AffineTransportNet


def test_recent_positive_subset_centering_and_future_invariance():
    rng=np.random.default_rng(852)
    stages=np.repeat([7.5,7.75,8.,9.],40)
    x=rng.uniform(.2,.8,(160,8)).astype(np.float32)
    x[rng.random(x.shape)<.2]=0
    donors=np.column_stack([x[80:120],np.full(40,.3)]).astype(np.float32)
    panel=[str(i) for i in range(8)]+['protected']
    coefficient=np.zeros((4,3));coefficient[-1]=.1
    net=AffineTransportNet(np.eye(8)[:3],np.zeros(8),coefficient)
    args=(8.,donors,panel,panel[:8],net,np.zeros(8,dtype=np.float32),np.ones(8,dtype=np.float32),np.arange(8),np.arange(8))
    model=RecentPositiveForecast(x,stages,*args,head_stages=1)
    recent=FeaturePanelForecast(x[stages==8.],stages[stages==8.],*args)
    globalfit=FeaturePanelForecast(x,stages,*args)
    np.testing.assert_array_equal(model.detection,globalfit.detection)
    np.testing.assert_array_equal(model.pmean,globalfit.pmean)
    np.testing.assert_array_equal(model.support,globalfit.support)
    np.testing.assert_array_equal(model.positive_coef,recent.positive_coef)
    predicted=model.predict(9.,'abundance',1.)[0]
    np.testing.assert_allclose(predicted,recent.predict(9.,'abundance',1.)[0],atol=1e-6)
    altered=x.copy();altered[stages>8.]=999
    other=RecentPositiveForecast(altered,stages,*args,head_stages=1)
    np.testing.assert_array_equal(predicted,other.predict(9.,'abundance',1.)[0])
    joint=model.predict(9.,'joint',1.,sampling='systematic')[0]
    np.testing.assert_array_equal(joint[:,-1],donors[:,-1])
    np.testing.assert_allclose(np.expm1(joint[:,:8]).sum(1),np.expm1(donors[:,:8]).sum(1),rtol=1e-6)
    assert model.audit['positive_fit_stages']==[8.]
