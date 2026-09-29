import numpy as np
from windowed_positive_head import WindowedPositiveForecast
from ridge_conditional_head import RidgeConditionalPositiveForecast
from transport_latent_flow import AffineTransportNet


def test_decoder_window_matches_exact_subset_and_excludes_unselected_stages():
    rng=np.random.default_rng(443);stages=np.repeat([7.5,7.75,8.,9.],32)
    x=rng.uniform(.2,.8,(128,8)).astype(np.float32);x[rng.random(x.shape)<.3]=0
    donors=np.column_stack([x[64:96],np.full(32,.3)]).astype(np.float32)
    panel=[str(i) for i in range(8)]+['protected'];coefficient=np.zeros((4,3));coefficient[-1]=.1
    net=AffineTransportNet(np.eye(8)[:3],np.zeros(8),coefficient)
    args=(8.,donors,panel,panel[:8],net,np.zeros(8,dtype=np.float32),np.ones(8,dtype=np.float32),np.arange(8))
    globalfit=WindowedPositiveForecast(x,stages,*args)
    control=RidgeConditionalPositiveForecast(x,stages,*args)
    np.testing.assert_array_equal(globalfit.positive_coef,control.positive_coef)
    model=WindowedPositiveForecast(x,stages,*args,head_stages=1)
    expected=RidgeConditionalPositiveForecast(x[stages==8.],stages[stages==8.],*args)
    np.testing.assert_array_equal(model.positive_coef,expected.positive_coef)
    assert model.audit['decoder_fit_stages']==[8.]
    changed=x.copy();changed[stages!=8.]=99
    other=WindowedPositiveForecast(changed,stages,*args,head_stages=1)
    np.testing.assert_array_equal(model.positive_coef,other.positive_coef)
    pred,_,audit=model.predict(9.,'abundance',.5)
    np.testing.assert_array_equal(pred,other.predict(9.,'abundance',.5)[0])
    np.testing.assert_array_equal(pred[:,-1],donors[:,-1]);np.testing.assert_array_equal(pred==0,donors==0)
    np.testing.assert_allclose(np.expm1(pred[:,:8]).sum(1),np.expm1(donors[:,:8]).sum(1),rtol=1e-6)
    assert audit['covariance_change_vs_reference']<=.4


def test_lazy_past_row_view_matches_materialized_gene_blocks():
    from windowed_positive_head import PastRowsView
    x=np.arange(60).reshape(10,6);rows=np.array([1,3,7]);view=PastRowsView(x,rows)
    np.testing.assert_array_equal(view[np.ix_([0,2],[1,4,5])],x[rows][np.ix_([0,2],[1,4,5])])
    assert view.shape==(3,6)
