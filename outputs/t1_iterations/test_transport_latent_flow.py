import numpy as np
from transport_latent_flow import log_transport,TemporalTransportFlow


def test_transport_matches_scalar_entropy_solution_and_balanced_marginals():
    p,audit=log_transport([[.3]],[.4],[.8],penalties=(1.,2.))
    expected=np.exp((np.log(.4)+2*np.log(.8)-.3)/3.05)
    np.testing.assert_allclose(p[0,0],expected,rtol=2e-6);assert audit['converged']
    p,audit=log_transport([[0.,.1],[.1,0.]],[.3,.7],[.6,.4],balanced=True)
    np.testing.assert_allclose(p.sum(1),[.3,.7],atol=1e-6)
    np.testing.assert_allclose(p.sum(0),[.6,.4],atol=1e-6);assert audit['converged']


def test_transport_fit_excludes_future_snapshots():
    rng=np.random.default_rng(191);x=rng.uniform(.1,1.,(128,8)).astype(np.float32)
    stages=np.repeat([7.5,7.75,8.,9.],32);rows=np.flatnonzero(stages<=8.)
    args=(stages,8.,[str(i) for i in range(8)],[str(i) for i in range(8)],np.arange(8),rows,np.zeros(len(rows)),'neutral')
    fitted=TemporalTransportFlow(x,*args,cell_budget=16);changed=x.copy();changed[stages>8]=99
    other=TemporalTransportFlow(changed,*args,cell_budget=16)
    np.testing.assert_array_equal(fitted.net.coefficient.numpy(),other.net.coefficient.numpy())
    assert fitted.audit['fit_max_stage']==8. and all(c['converged'] for c in fitted.audit['couplings'])
