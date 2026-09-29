import numpy as np
from nonlinear_positive_forecast import NonlinearPositiveForecast
from feature_panel_forecast import FeaturePanelForecast
from transport_latent_flow import AffineTransportNet


def test_nonlinear_zero_baseline_future_invariance_and_resume(tmp_path):
    rng=np.random.default_rng(701);stages=np.repeat([7.5,7.75,8.,9.],40)
    x=rng.uniform(.2,.8,(160,8)).astype(np.float32);x[rng.random(x.shape)<.3]=0
    donors=np.column_stack([x[80:120],np.full(40,.3)]).astype(np.float32)
    panel=[str(i) for i in range(8)]+['protected'];coefficient=np.zeros((4,3));coefficient[-1]=.1
    net=AffineTransportNet(np.eye(8)[:3],np.zeros(8),coefficient)
    args=(x,stages,8.,donors,panel,panel[:8],net,np.zeros(8,dtype=np.float32),np.ones(8,dtype=np.float32),np.arange(8),np.arange(8))
    zero=NonlinearPositiveForecast(*args,steps=0);control=FeaturePanelForecast(*args)
    for mode in ['abundance','joint']:
        np.testing.assert_array_equal(zero.predict(9.,mode,1.,sampling='systematic')[0],control.predict(9.,mode,1.,sampling='systematic')[0])
    checkpoint=tmp_path/'residual.pt'
    NonlinearPositiveForecast(*args,steps=2,checkpoint=checkpoint)
    resumed=NonlinearPositiveForecast(*args,steps=4,checkpoint=checkpoint,resume=True)
    direct=NonlinearPositiveForecast(*args,steps=4)
    for key,value in direct.residual_net.state_dict().items():np.testing.assert_array_equal(value.numpy(),resumed.residual_net.state_dict()[key].numpy())
    altered=x.copy();altered[stages>8.]=999
    other=NonlinearPositiveForecast(altered,*args[1:],steps=4)
    prediction=direct.predict(9.,'joint',1.,sampling='systematic')[0]
    np.testing.assert_array_equal(prediction,other.predict(9.,'joint',1.,sampling='systematic')[0])
    np.testing.assert_array_equal(direct.detection,control.detection)
    np.testing.assert_array_equal(prediction[:,-1],donors[:,-1])
    np.testing.assert_allclose(np.expm1(prediction[:,:8]).sum(1),np.expm1(donors[:,:8]).sum(1),rtol=1e-6)
