import numpy as np
import torch
from cnf_density_flow import DensityFlowNet
from feature_panel_forecast import FeaturePanelForecast
from anchor_head_calibration import AnchorHeadCalibration


def test_anchor_intercepts_keep_slopes_future_invariance_and_mass():
    rng=np.random.default_rng(482);stages=np.repeat([7.5,7.75,8.,9.],48)
    x=rng.uniform(.2,.8,(192,8)).astype(np.float32);x[rng.random(x.shape)<.2]=0
    donors=np.column_stack([x[96:144]*1.3,np.full(48,.3)]).astype(np.float32)
    panel=[str(i) for i in range(8)]+['protected']
    torch.manual_seed(482);net=DensityFlowNet(np.eye(8)[:3],np.zeros(8),8.,7.25)
    with torch.no_grad():net.field[-1].bias.fill_(.1)
    args=(stages,8.,donors,panel,panel[:8],net,np.zeros(8,dtype=np.float32),np.ones(8,dtype=np.float32),np.arange(8),np.array([0,1]))
    baseline=FeaturePanelForecast(x,*args);model=AnchorHeadCalibration(x,*args)
    np.testing.assert_array_equal(model.predict(9.,'joint',1.,sampling='systematic')[0],baseline.predict(9.,'joint',1.,sampling='systematic')[0])
    changed=x.copy();changed[stages>8.]=99;other=AnchorHeadCalibration(changed,*args)
    for key in ['anchor_positive_offset','anchor_detection_offset','positive_coef','detection']:
        np.testing.assert_array_equal(getattr(model,key),getattr(other,key))
    for positive,detection in [(.5,0.),(0.,1.),(1.,1.)]:
        model.configure(positive,detection);other.configure(positive,detection)
        pred=model.predict(9.,'joint',1.,sampling='systematic')[0]
        np.testing.assert_array_equal(pred,other.predict(9.,'joint',1.,sampling='systematic')[0])
        np.testing.assert_array_equal(pred[:,-1],donors[:,-1])
        np.testing.assert_allclose(np.expm1(pred[:,:8]).sum(1),np.expm1(donors[:,:8]).sum(1),rtol=1e-6)
        np.testing.assert_array_equal(model.predict(9.,'abundance',1.)[0],baseline.predict(9.,'abundance',1.)[0])
    assert np.max(np.abs(model.anchor_positive_offset))<=np.log(2.)
    assert np.max(np.abs(model.anchor_detection_offset))<=.25
