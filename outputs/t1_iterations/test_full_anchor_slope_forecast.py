import unittest
import numpy as np

from full_anchor_slope_forecast import fit_anchor_block
from ridge_conditional_head import conditional_ridge


class FullAnchorSlopeFitTest(unittest.TestCase):
    def test_block_matches_direct_conditional_and_detection_fits(self):
        rng = np.random.default_rng(27)
        h = rng.normal(size=(80, 3))
        detected = rng.random((80, 4)) < np.array([.7, .4, .9, 0.])
        response = detected * np.maximum(0, 1 + h @ rng.normal(size=(3, 4)))
        source_positive = rng.normal(size=(3, 4))
        source_detection = rng.normal(size=(3, 4))
        support = np.array([True, True, False, True])
        pdelta, ddelta, center, count, valid = fit_anchor_block(
            h, response, source_positive, source_detection, support)
        positive = (response > 0).astype(float)
        expected = conditional_ridge(h, positive, response)
        np.testing.assert_allclose(pdelta[:, valid], expected[:, valid] - source_positive[:, valid])
        np.testing.assert_array_equal(valid, (count >= 20) & support)
        np.testing.assert_array_equal(pdelta[:, ~valid], 0)
        np.testing.assert_array_equal(ddelta[:, ~valid], 0)
        np.testing.assert_allclose(center, h.T @ positive / np.maximum(count, 1))
        gram = (h-h.mean(0)).T @ (h-h.mean(0)) / len(h) + np.eye(h.shape[1])
        expected_detection = np.linalg.solve(gram, (h-h.mean(0)).T @ (positive-positive.mean(0)) / len(h))
        np.testing.assert_allclose(ddelta[:, valid], expected_detection[:, valid] - source_detection[:, valid])


if __name__ == '__main__':
    unittest.main()
