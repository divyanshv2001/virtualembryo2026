import numpy as np
from neural_ode_forecast import NeuralODEForecast


def test_neural_training_is_past_only_and_protects_output(tmp_path):
    rng=np.random.default_rng(191);stages=np.repeat([7.5,7.75,8.,9.],32)
    x=rng.uniform(.2,.8,(128,8)).astype(np.float32);x[::7,2]=0
    x[:,0]+=np.repeat([0.,.1,.2,.3],32)
    donors=np.column_stack([x[64:96],np.full(32,.3)]).astype(np.float32)
    panel=[str(i) for i in range(8)]+['protected']
    args=(stages,8.,donors,panel,panel[:8],np.arange(8))
    model=NeuralODEForecast(x,*args,beta=1.,pretrain_steps=2,joint_steps=2,batch=8,truth_batch=16)
    changed=x.copy();changed[stages>8]=99
    other=NeuralODEForecast(changed,*args,beta=1.,pretrain_steps=2,joint_steps=2,batch=8,truth_batch=16)
    np.testing.assert_array_equal(model.full_decoder,other.full_decoder)
    pred,indices,audit=model.predict(9.,.5)
    np.testing.assert_array_equal(pred,other.predict(9.,.5)[0])
    np.testing.assert_array_equal(indices,np.arange(32));np.testing.assert_array_equal(pred[:,8],donors[:,8])
    np.testing.assert_array_equal(pred==0,donors==0)
    np.testing.assert_allclose(np.expm1(pred[:,:8]).sum(1),np.expm1(donors[:,:8]).sum(1),rtol=1e-6)
    assert np.isfinite(pred).all() and np.any(pred!=donors)
    assert audit['covariance_change_vs_reference']<=.4
    checkpoint=tmp_path/'checkpoint.pt'
    fresh=NeuralODEForecast(x,*args,beta=0.,pretrain_steps=2,joint_steps=2,batch=8,truth_batch=16,checkpoint=checkpoint)
    resumed=NeuralODEForecast(x,*args,beta=0.,pretrain_steps=2,joint_steps=2,batch=8,truth_batch=16,checkpoint=checkpoint,resume=True)
    np.testing.assert_array_equal(fresh.predict(9.,.5)[0],resumed.predict(9.,.5)[0])
