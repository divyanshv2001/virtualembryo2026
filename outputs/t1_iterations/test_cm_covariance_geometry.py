"""Numerical/causal controls for the bounded Gaussian covariance audit."""
import numpy as np
from cm_covariance_geometry import bures_map,distance_squared,forecast,balanced_covariance,power


def main():
    rng=np.random.default_rng(20260928)
    a=rng.normal(size=(4,4));a=a@a.T+np.eye(4)
    b=rng.normal(size=(4,4));b=b@b.T+np.eye(4)
    t=bures_map(a,b)
    assert np.linalg.eigvalsh(t).min()>0
    assert np.allclose(t@a@t.T,b,atol=1e-9)
    assert np.allclose(bures_map(a,a),np.eye(4),atol=1e-9)
    assert distance_squared(a,a)<1e-9
    tiny=np.eye(4)*1e-6
    assert np.allclose(bures_map(tiny,tiny),np.eye(4),atol=1e-9)
    assert distance_squared(tiny,tiny)<1e-12
    assert abs(distance_squared(a,b)-distance_squared(b,a))<1e-9
    c,n,audit=forecast(a,b,7.5,7.75,8.)
    c2,n2,audit2=forecast(a,b,7.5,7.75,8.)
    assert np.array_equal(c,c2) and np.array_equal(n,n2) and audit==audit2
    assert np.linalg.eigvalsh(c).min()>0 and np.linalg.eigvalsh(n).min()>0
    try:power(np.diag([1.,-1.]),.5)
    except ValueError:pass
    else:raise AssertionError('Indefinite matrix accepted')
    try:forecast(a,b,7.5,7.75,7.75)
    except ValueError:pass
    else:raise AssertionError('Nonfuture target accepted')
    x=rng.normal(size=(20,4));groups=np.array(['a']*10+['b']*10)
    base,records=balanced_covariance(x,groups)
    shifted=x.copy();shifted[groups=='a']+=10;shifted[groups=='b']-=20
    assert np.allclose(base,balanced_covariance(shifted,groups)[0],atol=1e-12)
    assert all(r['included'] for r in records)
    duplicated=np.concatenate([x[:10],x[:10],x[10:]])
    labels=np.array(['a']*20+['b']*10)
    # Ledoit-Wolf shrinkage depends on sample count, even for duplicated rows.
    # Equal capture weighting is tested against independently estimated blocks.
    from sklearn.covariance import LedoitWolf
    expected=np.mean([LedoitWolf(assume_centered=True).fit(v-v.mean(0)).covariance_
                      for v in [duplicated[:20],duplicated[20:]]],axis=0)
    assert np.allclose(expected,balanced_covariance(duplicated,labels)[0],atol=1e-12)
    print('Covariance controls passed: OT map, identity, distance symmetry, PSD, deterministic null, forward time, capture centering/balance')


if __name__=='__main__':main()
