import numpy as np
import torch
from cnf_density_flow import DensityFlowNet
from feature_panel_forecast import FeaturePanelForecast
from anchor_slope_calibration import AnchorSlopeCalibration


def test_anchor_slopes_zero_replay_future_invariance_and_mass():
    rng=np.random.default_rng(482);stages=np.repeat([7.5,7.75,8.,9.],48)
    x=rng.uniform(.2,.8,(192,8)).astype(np.float32);x[rng.random(x.shape)<.2]=0
    donors=np.column_stack([x[96:144]*1.3,np.full(48,.3)]).astype(np.float32)
    panel=[str(i) for i in range(8)]+['protected']
    torch.manual_seed(482);net=DensityFlowNet(np.eye(8)[:3],np.zeros(8),8.,7.25)
    with torch.no_grad():net.field[-1].bias.fill_(.1)
    args=(stages,8.,donors,panel,panel[:8],net,np.zeros(8,dtype=np.float32),np.ones(8,dtype=np.float32),np.arange(8),np.array([0,1]))
    baseline=FeaturePanelForecast(x,*args);model=AnchorSlopeCalibration(x,*args)
    np.testing.assert_array_equal(model.predict(9.,'joint',1.,sampling='systematic')[0],baseline.predict(9.,'joint',1.,sampling='systematic')[0])
    changed=x.copy();changed[stages>8.]=99;other=AnchorSlopeCalibration(changed,*args)
    for key in ['anchor_positive_delta','anchor_detection_delta','positive_coef','detection']:
        np.testing.assert_array_equal(getattr(model,key),getattr(other,key))
    for positive,detection in [(.5,0.),(0.,1.),(1.,1.)]:
        model.configure(positive,detection);other.configure(positive,detection)
        pred=model.predict(9.,'joint',1.,sampling='systematic')[0]
        np.testing.assert_array_equal(pred,other.predict(9.,'joint',1.,sampling='systematic')[0])
        np.testing.assert_array_equal(pred[:,-1],donors[:,-1])
        np.testing.assert_allclose(np.expm1(pred[:,:8]).sum(1),np.expm1(donors[:,:8]).sum(1),rtol=1e-6)
        if positive==0.:np.testing.assert_array_equal(model.predict(9.,'abundance',1.)[0],baseline.predict(9.,'abundance',1.)[0])
    assert np.isfinite(model.anchor_positive_delta).all() and np.isfinite(model.anchor_detection_delta).all()
    assert np.any(model.anchor_positive_delta!=0)
    model.configure(1.,1.)
    anchor_mean=model.anchor_latent_mean[None,:]
    np.testing.assert_array_equal(model.detection_probability(anchor_mean,slice(None)),
                                  FeaturePanelForecast.detection_probability(model,anchor_mean,slice(None)))
    centers=model.anchor_positive_center.T
    difference=model.positive_log_value(centers,slice(None),9.)-FeaturePanelForecast.positive_log_value(model,centers,slice(None),9.)
    np.testing.assert_allclose(np.diag(difference),0.,atol=1e-12)
