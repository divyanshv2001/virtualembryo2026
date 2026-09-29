import numpy as np
from resolution_transport_flow import ResolutionTransportFlow
from transport_latent_flow import TemporalTransportFlow


def test_resolution_baseline_matches_and_all_pca_excludes_future():
    rng=np.random.default_rng(191);x=rng.uniform(.1,1.,(128,8)).astype(np.float32)
    stages=np.repeat([7.5,7.75,8.,9.],32);rows=np.flatnonzero(stages<=8.)
    args=(stages,8.,[str(i) for i in range(8)],[str(i) for i in range(8)],np.arange(8),rows,np.zeros(len(rows)),'neutral')
    baseline=TemporalTransportFlow(x,*args,cell_budget=16)
    fitted=ResolutionTransportFlow(x,*args,cell_budget=16)
    np.testing.assert_array_equal(baseline.net.coefficient.numpy(),fitted.net.coefficient.numpy())
    all_pca=ResolutionTransportFlow(x,*args,cell_budget=16,pca_budget=None)
    np.testing.assert_array_equal(all_pca.pca_rows,rows)
    for stage in fitted.sample_rows:
        np.testing.assert_array_equal(fitted.sample_rows[stage],all_pca.sample_rows[stage])
    changed=x.copy();changed[stages>8]=99
    other=ResolutionTransportFlow(changed,*args,cell_budget=16,pca_budget=None)
    np.testing.assert_array_equal(all_pca.net.coefficient.numpy(),other.net.coefficient.numpy())
