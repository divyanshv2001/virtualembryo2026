import unittest
import numpy as np
from tempfile import TemporaryDirectory
from pathlib import Path
from threadpoolctl import threadpool_limits
from detection_transfer import DetectionTransfer
from conditioned_generator import ConditionedGenerator as NonlinearTransport,mmd_loss_gradient,training_loss_gradient
from scipy.optimize import check_grad

class ConditionedTests(unittest.TestCase):
    def test_mmd_gradient(self):
        rng=np.random.default_rng(4);a=rng.normal(size=(7,3));b=rng.normal(size=(9,3));gammas=np.array([.2,1.])
        error=check_grad(lambda v:mmd_loss_gradient(v.reshape(7,3),b,gammas)[0],lambda v:mmd_loss_gradient(v.reshape(7,3),b,gammas)[1].ravel(),a.ravel())
        self.assertLess(error,1e-6)
        loss,gradient=mmd_loss_gradient(a,a,gammas)
        self.assertAlmostEqual(loss,0.,places=12)
        np.testing.assert_allclose(gradient,0.,atol=1e-12)
    def test_square_root_gradient(self):
        rng=np.random.default_rng(6);a=rng.normal(size=(7,3));b=rng.normal(size=(9,3));gammas=np.array([.2,1.])
        error=check_grad(lambda v:training_loss_gradient(v.reshape(7,3),b,gammas,'sqrt')[0],lambda v:training_loss_gradient(v.reshape(7,3),b,gammas,'sqrt')[1].ravel(),a.ravel())
        self.assertLess(error,1e-6)
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
            a=NonlinearTransport(x,stages,8.5,encoder,donors,base,indices,panel,symbols,steps=30,batch=16)
            b=NonlinearTransport(changed,stages,8.5,encoder,donors,base,indices,panel,symbols,steps=30,batch=16)
            np.testing.assert_array_equal(a.decoder,b.decoder)
            np.testing.assert_array_equal(a.probability_decoder,b.probability_decoder)
            np.testing.assert_array_equal(a.positive_decoder,b.positive_decoder)
            pred,origin,audit=a.predict(9.5,.25,mode='freeze',direction_gate=True,decoder='hurdle')
            np.testing.assert_array_equal(pred,b.predict(9.5,.25,mode='freeze',direction_gate=True,decoder='hurdle')[0])
            np.testing.assert_array_equal(pred[:,-1],base[:,-1])
            np.testing.assert_allclose(np.expm1(pred).sum(1),np.expm1(base).sum(1),rtol=1e-6)
            self.assertLessEqual(audit['covariance_change_vs_reference'],.4)
            # Force a known bounded detection increase to exercise new zero activation.
            a.parameters=[np.zeros_like(v) for v in a.parameters]
            a.probability_decoder[:]=0.;a.probability_decoder[-1,:]=.05
            a.probability_intercept[:]=.3;a.positive_decoder[:]=0.;a.positive_intercept[:]=.3
            activated,_,details=a.predict(9.5,1.,decoder='hurdle',detection_cap=.02)
            self.assertGreater(details['activated_values'],0)
            self.assertTrue(((base[:,:-1]==0)&(activated[:,:-1]>0)).any())
            np.testing.assert_array_equal(activated[:,-1],base[:,-1])
            np.testing.assert_allclose(np.expm1(activated).sum(1),np.expm1(base).sum(1),rtol=1e-6)
