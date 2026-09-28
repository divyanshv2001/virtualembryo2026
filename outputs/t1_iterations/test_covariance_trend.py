import numpy as np
import pytest
from copula_trend import CopulaTrend
from covariance_trend import CovarianceTrend,matrix_power_psd


def test_psd_power_is_finite_with_negative_forecast_eigenvalues():
    matrix=np.diag([-1.,2.]);root=matrix_power_psd(matrix,.5)
    assert np.isfinite(root).all() and np.linalg.eigvalsh(root).min()>0


@pytest.mark.parametrize('model_class',[CovarianceTrend,CopulaTrend])
def test_covariance_drift_excludes_future_and_preserves_full_panel_mass(model_class):
    rng=np.random.default_rng(115);stages=np.repeat([7.5,7.75,8.,9.],60);labels=np.repeat('A',240)
    x=rng.uniform(.2,.8,(240,8)).astype(np.float32)
    for j in range(4):x[j*60:(j+1)*60,1]=.3+(.2+.1*j)*x[j*60:(j+1)*60,0]
    x[::8,4]=0
    donors=np.column_stack([x[120:180],np.full(60,.3)]).astype(np.float32)
    panel=[str(i) for i in range(8)]+['protected'];args=(8.,donors,labels[120:180],panel,panel[:8],np.arange(8))
    model=model_class(x,stages,labels,*args)
    changed=x.copy();changed[stages>8]=99
    other=model_class(changed,stages,labels,*args)
    np.testing.assert_array_equal(model.cov_slope,other.cov_slope)
    np.testing.assert_array_equal(model.decoder,other.decoder)
    pred,indices,audit=(model.predict_copula(9.,1.) if model_class is CopulaTrend else model.predict_covariance(9.,.5))
    assert np.any(pred!=donors)
    np.testing.assert_array_equal(indices,np.arange(60));np.testing.assert_array_equal(pred[:,8],donors[:,8])
    np.testing.assert_array_equal(pred==0,donors==0)
    np.testing.assert_allclose(np.expm1(pred[:,:8]).sum(1),np.expm1(donors[:,:8]).sum(1),rtol=1e-6)
    assert audit['covariance_change_vs_reference']<=.4
