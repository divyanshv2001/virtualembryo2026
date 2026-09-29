import numpy as np
import torch
from cnf_mixture_base import fit_earliest_prior,MixtureBaseDensityFlowNet
from cnf_density_flow import inverse_density


def test_base_likelihood_agrees_with_sklearn_and_ignores_future():
    rng=np.random.default_rng(5);z=rng.normal(size=(80,2)).astype(np.float32);stages=np.repeat([7.5,7.75,8.,9.],20)
    for components in [1,8]:
        prior,model=fit_earliest_prior(z,stages,8.,components)
        changed=z.copy();changed[stages>8.]=np.nan
        repeated,_=fit_earliest_prior(changed,stages,8.,components)
        for k in ['weights','means','precision_cholesky']:np.testing.assert_array_equal(prior[k],repeated[k])
        net=MixtureBaseDensityFlowNet(np.eye(2),np.zeros(2),8.,7.25,prior)
        points=torch.tensor(z[:10],requires_grad=True)
        np.testing.assert_allclose(net.log_base(points).detach().numpy(),model.score_samples(z[:10]),rtol=2e-5,atol=2e-5)
        nll,energy=inverse_density(net.velocity,points,.5,noise=torch.ones_like(points),log_base=net.log_base)
        (nll.mean()+.1*energy.mean()).backward()
        assert torch.isfinite(nll).all() and all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.field.parameters())


def test_default_base_replay_and_prior_checkpoint_guard(tmp_path):
    from cnf_manifold_flow import train_manifold_density
    import pytest
    rng=np.random.default_rng(5);z=rng.normal(size=(24,2)).astype(np.float32);stages=np.repeat([7.5,7.75],12)
    prior,_=fit_earliest_prior(z,stages,7.75,1)
    net=MixtureBaseDensityFlowNet(np.eye(2),np.zeros(2),7.75,7.25,prior)
    values=torch.tensor(z[:4]);noise=torch.ones_like(values)
    first=inverse_density(net.velocity,values.clone(),.5,noise=noise)
    explicit=inverse_density(net.velocity,values.clone(),.5,noise=noise,log_base=lambda t:-.5*(t.square()+np.log(2*np.pi)).sum(1))
    for a,c in zip(first,explicit):torch.testing.assert_close(a,c,rtol=0,atol=0)
    path=tmp_path/'prior.pt'
    train_manifold_density(net,z,stages,.1,path,lambda *a,**kw:None,steps=2,density_weight=10.)
    assert torch.load(path,weights_only=False)['base_signature']==net.base_signature
    changed={**prior,'means':prior['means']+.1}
    other=MixtureBaseDensityFlowNet(np.eye(2),np.zeros(2),7.75,7.25,changed)
    with pytest.raises(ValueError,match='Checkpoint configuration'):
        train_manifold_density(other,z,stages,.1,path,lambda *a,**kw:None,resume=True,steps=2,density_weight=10.)


def test_observed_origin_zero_duration_equals_frozen_base_density():
    z=np.random.default_rng(6).normal(size=(24,2)).astype(np.float32);stages=np.repeat([7.5,7.75],12)
    prior,_=fit_earliest_prior(z,stages,7.75,1)
    net=MixtureBaseDensityFlowNet(np.eye(2),np.zeros(2),7.75,7.5,prior)
    with torch.no_grad():net.field[-1].weight.normal_(0,.03)
    values=torch.tensor(z[:4])
    expected=-net.log_base(values)
    nll,energy=inverse_density(net.velocity,values.clone(),0.,noise=torch.ones_like(values),log_base=net.log_base)
    torch.testing.assert_close(nll,expected,rtol=0,atol=0)
    torch.testing.assert_close(energy,torch.zeros(4),rtol=0,atol=0)
    (nll.mean()+.1*energy.mean()).backward()
    for parameter in net.field.parameters():
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad,torch.zeros_like(parameter),rtol=0,atol=0)
