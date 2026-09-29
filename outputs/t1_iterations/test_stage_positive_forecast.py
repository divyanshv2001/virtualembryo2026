import numpy as np
from stage_positive_forecast import StagePositiveForecast
from feature_panel_forecast import FeaturePanelForecast
from ridge_conditional_head import conditional_ridge
from transport_latent_flow import AffineTransportNet


def test_stage_positive_design_matches_conditional_fit_and_excludes_future():
    rng=np.random.default_rng(943)
    stages=np.repeat([7.5,7.75,8.,9.],40)
    x=rng.uniform(.2,.8,(160,8)).astype(np.float32);x[rng.random(x.shape)<.2]=0
    donors=np.column_stack([x[80:120],np.full(40,.3)]).astype(np.float32)
    panel=[str(i) for i in range(8)]+['protected'];coefficient=np.zeros((4,3));coefficient[-1]=.1
    net=AffineTransportNet(np.eye(8)[:3],np.zeros(8),coefficient)
    args=(x,stages,8.,donors,panel,panel[:8],net,np.zeros(8,dtype=np.float32),np.ones(8,dtype=np.float32),np.arange(8),np.arange(8))
    for design in ['additive','interaction']:
        model=StagePositiveForecast(*args,design=design)
        h=x[:120,:3].astype(float)-model.zcenter
        matrix=model.design_matrix(h,stages[:120]);detected=(x[:120]>0).astype(float)
        response=np.log(np.maximum(np.expm1(x[:120].astype(float)),1e-8))*detected
        np.testing.assert_allclose(model.stage_coef,conditional_ridge(matrix,detected,response),atol=1e-7)
        globalfit=FeaturePanelForecast(*args)
        np.testing.assert_array_equal(model.detection,globalfit.detection)
        changed=x.copy();changed[stages>8.]=999
        other=StagePositiveForecast(changed,*args[1:],design=design)
        pred=model.predict(9.,'joint',1.,sampling='systematic')[0]
        np.testing.assert_array_equal(pred,other.predict(9.,'joint',1.,sampling='systematic')[0])
        np.testing.assert_array_equal(pred[:,-1],donors[:,-1])
        np.testing.assert_allclose(np.expm1(pred[:,:8]).sum(1),np.expm1(donors[:,:8]).sum(1),rtol=1e-6)
