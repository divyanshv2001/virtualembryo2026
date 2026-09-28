import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from robust_population import RobustPopulation, stable_slope


class RobustTests(unittest.TestCase):
    def test_reversals_and_uncertainty_reduce_trends(self):
        times = np.array([0., 1., 2.])
        means = np.array([[0., 0.], [1., 2.], [2., 1.]])
        slope = stable_slope(times, means, np.zeros_like(means))
        self.assertGreater(slope[0], .99); self.assertEqual(slope[1], 0)
        uncertain = stable_slope(times, means, np.full_like(means, 10))
        self.assertLess(uncertain[0], .1)

    def test_future_invariance_support_and_covariance_budget(self):
        rng = np.random.default_rng(7)
        counts = rng.poisson(.7, (240, 40)).astype(float)
        x = np.log1p(counts*10000/counts.sum(1)[:, None]).astype(np.float32)
        stages = np.repeat([7., 7.25, 7.5, 8.5], 60)
        changed = x.copy(); changed[stages > 7.5] = 100
        with threadpool_limits(limits=2):
            a = RobustPopulation(x, stages, 7.5, states=4, donor_cap=40)
            b = RobustPopulation(changed, stages, 7.5, states=4, donor_cap=40)
            prediction, donor, _, weights = a.predict(8.5, 100., 100.)
            np.testing.assert_allclose(prediction, b.predict(8.5, 100., 100.)[0])
            np.testing.assert_array_equal(prediction == 0, donor == 0)
            np.testing.assert_allclose(np.expm1(prediction).sum(1), 10000, rtol=1e-6)
            self.assertGreaterEqual(1/np.sum(weights**2), .8*len(weights))
            self.assertLessEqual(a.last_audit['covariance_change_vs_reference'], .1)
            copy, _, _, _ = a.predict(8.5)
            np.testing.assert_array_equal(copy, a.donor)


if __name__ == '__main__': unittest.main()
