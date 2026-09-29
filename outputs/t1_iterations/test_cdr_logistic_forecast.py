import numpy as np
import ast
from pathlib import Path
from cdr_logistic_forecast import CDRLogisticForecast
from logistic_detection_head import fit_logistic
from feature_panel_forecast import FeaturePanelForecast
from transport_latent_flow import AffineTransportNet


def test_cdr_runner_plan_uses_json_serializable_keys():
    tree=ast.parse(Path(__file__).with_name('cnf_cdr_logistic_challenge.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name) and node.value.id=='plan':
            assert not isinstance(node.slice,ast.Tuple), 'Plan dictionary keys must be strings'


def test_cdr_logistic_matches_optimizer_and_keeps_positive_head_past_only():
    rng=np.random.default_rng(512)
    stages=np.repeat([7.5,7.75,8.,9.],40)
    x=rng.uniform(.2,.8,(160,8)).astype(np.float32);x[rng.random(x.shape)<.3]=0
    donors=np.column_stack([x[80:120],np.full(40,.3)]).astype(np.float32)
    panel=[str(i) for i in range(8)]+['protected'];coefficient=np.zeros((4,3));coefficient[-1]=.1
    net=AffineTransportNet(np.eye(8)[:3],np.zeros(8),coefficient)
    args=(x,stages,8.,donors,panel,panel[:8],net,np.zeros(8,dtype=np.float32),np.ones(8,dtype=np.float32),np.arange(8),np.arange(8))
    control=FeaturePanelForecast(*args)
    for ridge in [.1,1.]:
        model=CDRLogisticForecast(*args,detection_ridge=ridge)
        h=x[:120,:3].astype(float)-model.zcenter;cdr=(x[:120]>0).mean(1)
        matrix=np.column_stack([h,(cdr-model.cdr_mean)/model.cdr_scale])
        coef,_=fit_logistic(matrix,(x[:120]>0).astype(float),ridge)
        np.testing.assert_array_equal(model.cdr_logistic_coef,coef)
        np.testing.assert_array_equal(model.predict(9.,'abundance',1.)[0],control.predict(9.,'abundance',1.)[0])
        changed=x.copy();changed[stages>8.]=999
        other=CDRLogisticForecast(changed,*args[1:],detection_ridge=ridge)
        pred=model.predict(9.,'joint',1.,sampling='systematic')[0]
        np.testing.assert_array_equal(pred,other.predict(9.,'joint',1.,sampling='systematic')[0])
        np.testing.assert_array_equal(pred[:,-1],donors[:,-1])
        np.testing.assert_allclose(np.expm1(pred[:,:8]).sum(1),np.expm1(donors[:,:8]).sum(1),rtol=1e-6)
