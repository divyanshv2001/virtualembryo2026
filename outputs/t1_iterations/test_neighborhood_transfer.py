import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from neighborhood_transfer import NeighborhoodTransfer


class NeighborhoodTests(unittest.TestCase):
    def test_past_only_sparse_activation_and_protected_mass(self):
        rng = np.random.default_rng(8)
        stages = np.repeat([8.,8.25,8.5,9.5],60)
        x = np.log1p(rng.poisson(.8,(240,40))).astype(np.float32)
        changed = x.copy(); changed[stages > 8.5] = 100
        symbols = [f'G{i}' for i in range(40)]
        official = symbols+['protected']
        donors = np.column_stack([x[stages == 8.5],np.ones(60,dtype=np.float32)])
        with threadpool_limits(limits=2):
            a = NeighborhoodTransfer(x,stages,8.5,donors,official,symbols)
            b = NeighborhoodTransfer(changed,stages,8.5,donors,official,symbols)
            np.testing.assert_array_equal(a.local_slope,b.local_slope)
            for activations in [0,2]:
                pred,indices,audit = a.predict_neighborhood(9.5,.5,activations)
                np.testing.assert_array_equal(pred[:,-1],donors[indices,-1])
                self.assertTrue((((donors == 0) & (pred > 0)).sum(1) <= activations).all())
                np.testing.assert_allclose(np.expm1(pred).sum(1),np.expm1(donors).sum(1),rtol=1e-6)
                self.assertTrue(np.isfinite(pred).all()); self.assertTrue((pred >= 0).all())


if __name__ == '__main__': unittest.main()
