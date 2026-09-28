import numpy as np
from annotation_trend import AnnotationTrend


def test_future_expression_and_labels_cannot_change_forecast():
    rng=np.random.default_rng(72);stages=np.repeat([7.5,7.75,8.,9.],80)
    labels=np.tile(np.repeat(['A','B'],40),4)
    x=rng.uniform(.3,.5,(320,3)).astype(np.float32)
    for i in range(4):
        x[i*80:(i+1)*80,0]+=.15*i
        for j in range(2):
            x[i*80+j*40+10+5*i:i*80+(j+1)*40,1]=0
    donors=np.column_stack([x[160:240],np.full(80,.2)]).astype(np.float32)
    args=(8.,donors,labels[160:240],['G0','G1','G2','protected'],['G0','G1','G2'],[0,1,2])
    model=AnnotationTrend(x,stages,labels,*args)
    changed=x.copy();changed[stages>8.]=99
    altered=labels.copy();altered[stages>8.]='future_only'
    other=AnnotationTrend(changed,stages,altered,*args)
    np.testing.assert_array_equal(model.positive_slope,other.positive_slope)
    np.testing.assert_array_equal(model.detection_slope,other.detection_slope)
    pred,indices,audit=model.predict(9.)
    other_pred,_,_=other.predict(9.)
    np.testing.assert_array_equal(pred,other_pred)
    assert np.any(pred!=donors) and audit['actual_additions']>0
    np.testing.assert_array_equal(pred[:,3],donors[:,3])
    np.testing.assert_array_equal(indices,np.arange(80))
    np.testing.assert_allclose(np.expm1(pred[:,:3]).sum(1),np.expm1(donors[:,:3]).sum(1),rtol=1e-6)
    assert audit['covariance_change_vs_reference']<=.4
    unsupported=AnnotationTrend(x,stages,labels,*args,min_source_cells=100,min_positive_cells=20)
    persistence,_,unsupported_audit=unsupported.predict(9.)
    np.testing.assert_array_equal(persistence,donors)
    assert unsupported_audit['supported_donor_fraction']==0
