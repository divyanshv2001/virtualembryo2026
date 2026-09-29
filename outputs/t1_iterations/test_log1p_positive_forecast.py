import numpy as np
import torch
from cnf_density_flow import DensityFlowNet
from log1p_positive_forecast import Log1pPositiveForecast
from anchor_slope_calibration import AnchorSlopeCalibration


def test_log1p_response_future_exclusion_no_motion_and_output_guards():
    rng=np.random.default_rng(482);stages=np.repeat([7.5,7.75,8.,9.],48)
    x=rng.uniform(.2,.8,(192,8)).astype(np.float32);x[rng.random(x.shape)<.2]=0
    donors=np.column_stack([x[96:144]*1.3,np.full(48,.3)]).astype(np.float32)
    panel=[str(i) for i in range(8)]+['protected']
    torch.manual_seed(482);net=DensityFlowNet(np.eye(8)[:3],np.zeros(8),8.,7.25)
    args=(stages,8.,donors,panel,panel[:8],net,np.zeros(8,dtype=np.float32),np.ones(8,dtype=np.float32),np.arange(8),np.array([0,1]))
    model=Log1pPositiveForecast(x,*args);baseline=AnchorSlopeCalibration(x,*args)
    np.testing.assert_array_equal(model.detection,baseline.detection)
    np.testing.assert_array_equal(model.anchor_detection_delta,baseline.anchor_detection_delta)
    past=x[stages<=8.];expected=past.sum(0)/np.maximum((past>0).sum(0),1)
    np.testing.assert_allclose(model.positive_mean,expected,rtol=1e-6)
    for alpha in [0.,.5]:
        model.configure(alpha,.5)
        np.testing.assert_allclose(model.predict(9.,'abundance',1.)[0],donors,atol=1e-7)
    with torch.no_grad():net.field[-1].bias.fill_(.1)
    changed=x.copy();changed[stages>8.]=99;other=Log1pPositiveForecast(changed,*args)
    for key in ['positive_mean','positive_coef','anchor_positive_delta']:
        np.testing.assert_array_equal(getattr(model,key),getattr(other,key))
    for alpha in [0.,.5]:
        model.configure(alpha,.5);other.configure(alpha,.5)
        prediction=model.predict(9.,'joint',1.,sampling='systematic')[0]
        np.testing.assert_array_equal(prediction,other.predict(9.,'joint',1.,sampling='systematic')[0])
        assert np.isfinite(prediction).all() and (prediction>=0).all()
        np.testing.assert_array_equal(prediction[:,-1],donors[:,-1])
        np.testing.assert_allclose(np.expm1(prediction[:,:8]).sum(1),np.expm1(donors[:,:8]).sum(1),rtol=1e-6)
