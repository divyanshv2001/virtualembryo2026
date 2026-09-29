import numpy as np
from cdr_detection_forecast import CDRDetectionForecast
from feature_panel_forecast import FeaturePanelForecast
from transport_latent_flow import AffineTransportNet


def test_cdr_matches_ridge_and_preserves_abundance_and_future_invariance():
    rng=np.random.default_rng(389)
    stages=np.repeat([7.5,7.75,8.,9.],40)
    x=rng.uniform(.2,.8,(160,8)).astype(np.float32);x[rng.random(x.shape)<.3]=0
    donors=np.column_stack([x[80:120],np.full(40,.3)]).astype(np.float32)
    panel=[str(i) for i in range(8)]+['protected'];coefficient=np.zeros((4,3));coefficient[-1]=.1
    net=AffineTransportNet(np.eye(8)[:3],np.zeros(8),coefficient)
    args=(x,stages,8.,donors,panel,panel[:8],net,np.zeros(8,dtype=np.float32),np.ones(8,dtype=np.float32),np.arange(8),np.arange(8))
    control=FeaturePanelForecast(*args)
    for ridge in [.1,1.]:
        model=CDRDetectionForecast(*args,cdr_ridge=ridge)
        cdr=(x[:120]>0).mean(1);h=x[:120,:3].astype(float)-model.zcenter
        matrix=np.column_stack([h,(cdr-cdr.mean())/max(cdr.std(),1e-3)])
        expected=np.linalg.solve(matrix.T@matrix/120+np.diag([1.,1.,1.,ridge]),matrix.T@(x[:120]>0)/120)
        np.testing.assert_allclose(model.cdr_coef,expected,atol=1e-8)
        np.testing.assert_array_equal(model.positive_coef,control.positive_coef)
        np.testing.assert_array_equal(model.predict(9.,'abundance',1.)[0],control.predict(9.,'abundance',1.)[0])
        changed=x.copy();changed[stages>8.]=999
        other=CDRDetectionForecast(changed,*args[1:],cdr_ridge=ridge)
        pred=model.predict(9.,'joint',1.,sampling='systematic')[0]
        np.testing.assert_array_equal(pred,other.predict(9.,'joint',1.,sampling='systematic')[0])
        np.testing.assert_array_equal(pred[:,-1],donors[:,-1])
        np.testing.assert_allclose(np.expm1(pred[:,:8]).sum(1),np.expm1(donors[:,:8]).sum(1),rtol=1e-6)
