import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from detection_transfer import DetectionTransfer


class DetectionTests(unittest.TestCase):
    def test_future_exclusion_and_support_changes_conserve_mass(self):
        rng = np.random.default_rng(77)
        stages = np.repeat([8., 8.25, 8.5, 9.5], 160)
        counts = rng.poisson(2, (640, 40)).astype(float)
        for j,t in enumerate([8., 8.25, 8.5]):
            counts[stages == t, 0] = (rng.random(160) < [.1,.3,.5][j])*3
            counts[stages == t, 1] = (rng.random(160) < [.9,.7,.5][j])*3
        x = np.log1p(counts).astype(np.float32)
        symbols = [f'G{i}' for i in range(40)]
        official = symbols+['protected']
        donors = np.column_stack([x[stages == 8.5], np.ones(160, dtype=np.float32)])
        future = x.copy(); future[stages > 8.5] = 100
        with threadpool_limits(limits=2):
            a = DetectionTransfer(x, stages, 8.5, donors, official, symbols, states=1, alignment='identity')
            b = DetectionTransfer(future, stages, 8.5, donors, official, symbols, states=1, alignment='identity')
            np.testing.assert_array_equal(a.detection_slope, b.detection_slope)
            result, indices, audit = a.predict_detection(9.5, 1., 0., .02)
            np.testing.assert_array_equal(result, b.predict_detection(9.5, 1., 0., .02)[0])
            np.testing.assert_array_equal(result[:,-1], donors[indices,-1])
            np.testing.assert_allclose(np.expm1(result).sum(1), np.expm1(donors[indices]).sum(1), rtol=1e-6)
            rejected, _, rejected_audit = a.predict_detection(9.5, 1., 0., .05)
            self.assertTrue(rejected_audit['detection_guard_rejected'])
            np.testing.assert_array_equal(rejected, a.predict(9.5, 'state', .5, 0.)[0])
            self.assertGreater(audit['actual_additions'], 0)
            self.assertGreater(audit['actual_deletions'], 0)
            self.assertTrue(np.isfinite(result).all()); self.assertTrue((result >= 0).all())


if __name__ == '__main__': unittest.main()
