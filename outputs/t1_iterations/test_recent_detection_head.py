import numpy as np
import torch
from cnf_density_flow import DensityFlowNet
from feature_panel_forecast import FeaturePanelForecast
from recent_detection_head import RecentDetectionForecast


def test_recent_detection_changes_only_probability_fit_and_remains_past_only():
    rng=np.random.default_rng(511);stages=np.repeat([7.5,7.75,8.,9.],32)
    x=rng.uniform(.2,.8,(128,8)).astype(np.float32);x[rng.random(x.shape)<.3]=0
    donors=np.column_stack([x[64:96],np.full(32,.3)]).astype(np.float32);panel=[str(i) for i in range(8)]+['protected']
    torch.manual_seed(511);net=DensityFlowNet(np.eye(8)[:3],np.zeros(8),8.,7.25)
    with torch.no_grad():net.field[-1].bias.fill_(.1)
    args=(stages,8.,donors,panel,panel[:8],net,np.zeros(8,dtype=np.float32),np.ones(8,dtype=np.float32),np.arange(8),np.array([0,1]))
    base=FeaturePanelForecast(x,*args);recent=RecentDetectionForecast(x,*args,head_stages=2)
    changed=x.copy();changed[stages>8.]=99;repeat=RecentDetectionForecast(changed,*args,head_stages=2)
    np.testing.assert_array_equal(recent.positive_coef,base.positive_coef)
    np.testing.assert_array_equal(recent.predict(9.,'abundance',1.)[0],base.predict(9.,'abundance',1.)[0])
    np.testing.assert_array_equal(recent.recent_detection,repeat.recent_detection)
    selected=np.flatnonzero(np.isin(stages,[7.75,8.]))
    subset=FeaturePanelForecast(x[selected],*tuple([stages[selected]]+list(args[1:])))
    h=np.zeros((4,3));expected=subset.detection_probability(h+base.zcenter-subset.zcenter,slice(None))
    np.testing.assert_allclose(recent.detection_probability(h,slice(None)),expected)
    prediction,_,audit=recent.predict(9.,'joint',1.,sampling='systematic')
    np.testing.assert_array_equal(prediction[:,-1],donors[:,-1])
    np.testing.assert_allclose(np.expm1(prediction[:,:8]).sum(1),np.expm1(donors[:,:8]).sum(1),rtol=1e-6)
    assert audit['recent_detection_stages']==[7.75,8.]
