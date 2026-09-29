import numpy as np
from conditional_positive_head import conditional_ridge
from conditional_positive_head import FullConditionalPositiveForecast
from transport_latent_flow import AffineTransportNet


def test_conditional_ridge_matches_direct_centered_gene_fits():
    rng=np.random.default_rng(428);h=rng.normal(size=(90,4));h[:,1]+=.8*h[:,0]
    detected=(rng.random((90,3))>.25).astype(float)
    y=(h@rng.normal(size=(4,3))+rng.normal(size=(90,3)))*detected
    coefficient=conditional_ridge(h,detected,y)
    for j in range(3):
        used=detected[:,j]>0;z=h[used]-h[used].mean(0);v=y[used,j]-y[used,j].mean()
        expected=np.linalg.solve(z.T@z/used.sum()+np.eye(4),z.T@v/used.sum())
        np.testing.assert_allclose(coefficient[:,j],expected,rtol=1e-10,atol=1e-10)
    detected[:]=0;detected[:19]=1
    np.testing.assert_array_equal(conditional_ridge(h,detected,y*detected),np.zeros((4,3)))


def test_full_conditional_fit_excludes_future_and_preserves_output_guards():
    rng=np.random.default_rng(430);stages=np.repeat([7.5,7.75,8.,9.],32)
    x=rng.uniform(.2,.8,(128,8)).astype(np.float32);x[rng.random(x.shape)<.3]=0
    donors=np.column_stack([x[64:96],np.full(32,.3)]).astype(np.float32)
    panel=[str(i) for i in range(8)]+['protected']
    coefficient=np.zeros((4,3));coefficient[-1]=.05
    net=AffineTransportNet(np.eye(8)[:3],np.zeros(8),coefficient)
    args=(stages,8.,donors,panel,panel[:8],net,np.zeros(8,dtype=np.float32),np.ones(8,dtype=np.float32),np.arange(8))
    model=FullConditionalPositiveForecast(x,*args);changed=x.copy();changed[stages>8]=99
    other=FullConditionalPositiveForecast(changed,*args)
    np.testing.assert_array_equal(model.positive_coef,other.positive_coef)
    pred,_,audit=model.predict(9.,'abundance',.5)
    np.testing.assert_array_equal(pred,other.predict(9.,'abundance',.5)[0])
    np.testing.assert_array_equal(pred[:,-1],donors[:,-1])
    np.testing.assert_array_equal(pred==0,donors==0)
    np.testing.assert_allclose(np.expm1(pred[:,:8]).sum(1),np.expm1(donors[:,:8]).sum(1),rtol=1e-6)
    assert audit['covariance_change_vs_reference']<=.4
