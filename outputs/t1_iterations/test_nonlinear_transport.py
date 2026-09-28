import unittest
import numpy as np
from tempfile import TemporaryDirectory
from pathlib import Path
from threadpoolctl import threadpool_limits
from detection_transfer import DetectionTransfer
from nonlinear_transport import coupling,NonlinearTransport

class NonlinearTests(unittest.TestCase):
    def test_coupling_marginals(self):
        rng=np.random.default_rng(4);a=rng.normal(size=(20,4));b=rng.normal(size=(25,4))
        plan,error=coupling(a,b)
        self.assertLess(error,1e-5)
        np.testing.assert_allclose(plan.sum(1),1/20,atol=1e-5)
        np.testing.assert_allclose(plan.sum(0),1/25,atol=1e-5)
    def test_future_exclusion_mapping_and_mass(self):
        rng=np.random.default_rng(5);stages=np.repeat([7.5,7.75,8.,8.25,8.5,9.5],60)
        x=np.log1p(rng.poisson(2,(360,20))).astype(np.float32);changed=x.copy();changed[stages>8.5]=100
        symbols=[f'G{i}' for i in range(20)];donors=x[stages==8.5].copy();panel=symbols+['protected']
        donors=np.column_stack([donors,np.full(len(donors),.2,dtype=np.float32)])
        with threadpool_limits(limits=2),TemporaryDirectory() as directory:
            m=DetectionTransfer(x,stages,8.5,donors,panel,symbols,alignment='identity',feature_scaling='unit',covariance_limit=.4)
            path=Path(directory)/'m.npz';m.save(path)
            with np.load(path) as f:encoder={k:f[k] for k in f.files}
            base,indices,_=m.predict_detection(9.5,.5,1.,.02)
            a=NonlinearTransport(x,stages,8.5,encoder,donors,base,indices,panel,symbols,pair_cells=40)
            b=NonlinearTransport(changed,stages,8.5,encoder,donors,base,indices,panel,symbols,pair_cells=40)
            np.testing.assert_array_equal(a.decoder,b.decoder)
            pred,origin,audit=a.predict(9.5,.25)
            np.testing.assert_array_equal(pred,b.predict(9.5,.25)[0])
            np.testing.assert_array_equal(pred[:,-1],base[:,-1])
            np.testing.assert_allclose(np.expm1(pred).sum(1),np.expm1(base).sum(1),rtol=1e-6)
            self.assertLessEqual(audit['covariance_change_vs_reference'],.4)
