import numpy as np
from recent_transport_flow import RecentTransportFlow
from dimensional_transport_flow import DimensionalTransportFlow


def test_recent_window_global_control_matches_and_future_is_excluded():
    rng=np.random.default_rng(441);x=rng.uniform(.1,1.,(160,8)).astype(np.float32)
    stages=np.repeat([7.5,7.75,8.,8.25,9.],32);rows=np.flatnonzero(stages<=8.25)
    args=(stages,8.25,[str(i) for i in range(8)],[str(i) for i in range(8)],np.arange(8),rows,np.zeros(len(rows)),'neutral')
    baseline=DimensionalTransportFlow(x,*args,cell_budget=16,latent_dim=8)
    control=RecentTransportFlow(x,*args,cell_budget=16,latent_dim=8)
    np.testing.assert_array_equal(baseline.net.coefficient.numpy(),control.net.coefficient.numpy())
    for window in [1,2]:
        model=RecentTransportFlow(x,*args,cell_budget=16,latent_dim=8,recent_intervals=window)
        assert len(model.audit['velocity_fit_intervals'])==window
        assert model.audit['velocity_fit_intervals'][-1]==[8.,8.25]
        np.testing.assert_array_equal(model.net.basis.numpy(),baseline.net.basis.numpy())
        for stage in control.sample_rows:np.testing.assert_array_equal(model.sample_rows[stage],baseline.sample_rows[stage])
        changed=x.copy();changed[stages>8.25]=99
        other=RecentTransportFlow(changed,*args,cell_budget=16,latent_dim=8,recent_intervals=window)
        np.testing.assert_array_equal(model.net.coefficient.numpy(),other.net.coefficient.numpy())
        assert not np.array_equal(model.net.coefficient.numpy(),control.net.coefficient.numpy())
