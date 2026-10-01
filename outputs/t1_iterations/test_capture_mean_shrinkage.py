import numpy as np
from capture_mean_shrinkage import forecast


def main():
    a=np.array([[1.,2.,3.],[1.,2.,3.]])
    b=a+np.array([.5,-.5,1.])
    p,audit=forecast(a,b)
    np.testing.assert_allclose(p['capture_shrunk'],p['capture_linear'])
    np.testing.assert_allclose(forecast(a[::-1],b[::-1])[0]['capture_shrunk'],p['capture_shrunk'])
    np.testing.assert_allclose(forecast(a+7,b+7)[0]['capture_shrunk'],p['capture_shrunk']+7)
    np.testing.assert_allclose(forecast(a,a)[0]['capture_shrunk'],a.mean(0))
    # Large between-capture disagreement must eliminate the estimated signal prior.
    noisy=np.array([[0.,0.,0.],[4.,4.,4.]])
    p,audit=forecast(noisy,noisy+.1)
    assert audit['tau2']==0
    np.testing.assert_allclose(p['capture_shrunk'],p['persistence'])
    for invalid in [np.ones((1,3)),np.full((2,3),np.nan)]:
        try:forecast(invalid,b)
        except ValueError:pass
        else:raise AssertionError('Invalid input accepted')
    print('Capture mean shrinkage numerical controls passed')


if __name__=='__main__':main()
