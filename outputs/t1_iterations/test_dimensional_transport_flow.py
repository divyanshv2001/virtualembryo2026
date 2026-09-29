import numpy as np
from dimensional_transport_flow import DimensionalTransportFlow
from resolution_transport_flow import ResolutionTransportFlow


def test_dimension_control_matches_and_five_dimensional_fit_excludes_future():
    rng=np.random.default_rng(191);x=rng.uniform(.1,1.,(128,8)).astype(np.float32)
    stages=np.repeat([7.5,7.75,8.,9.],32);rows=np.flatnonzero(stages<=8.)
    args=(stages,8.,[str(i) for i in range(8)],[str(i) for i in range(8)],np.arange(8),rows,np.zeros(len(rows)),'neutral')
    baseline=ResolutionTransportFlow(x,*args,cell_budget=16)
    control=DimensionalTransportFlow(x,*args,cell_budget=16,latent_dim=16)
    np.testing.assert_array_equal(baseline.net.coefficient.numpy(),control.net.coefficient.numpy())
    fitted=DimensionalTransportFlow(x,*args,cell_budget=16,latent_dim=5)
    assert fitted.net.coefficient.shape==(6,5)
    changed=x.copy();changed[stages>8]=99
    other=DimensionalTransportFlow(changed,*args,cell_budget=16,latent_dim=5)
    np.testing.assert_array_equal(fitted.net.coefficient.numpy(),other.net.coefficient.numpy())
    for stage in control.sample_rows:
        np.testing.assert_array_equal(control.sample_rows[stage],fitted.sample_rows[stage])
