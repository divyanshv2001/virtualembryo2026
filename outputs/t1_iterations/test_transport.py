import unittest
import numpy as np
from scipy import sparse
from local_transport import select_features, latent_space, regularized_log_ratio, apply_ratio


class TransportTests(unittest.TestCase):
    def test_no_change_is_identity(self):
        x = np.array([[0., .2, 2.]], dtype=np.float32)
        ratio = regularized_log_ratio(np.ones(3), np.ones(3), np.ones(3))
        np.testing.assert_array_equal(apply_ratio(x, ratio), x)

    def test_uncertain_changes_shrink_and_multiplier_is_bounded(self):
        a = np.array([1., 1., 1.])
        b = np.array([10., 10., .001])
        r = regularized_log_ratio(a, b, np.array([0., 1e9, 0.]))
        self.assertLess(abs(r[1]), abs(r[0]))
        self.assertTrue(np.all(abs(r) <= np.log(2)))
        x = np.array([0., 1., 2.])
        y = apply_ratio(x, np.array([0., r[0], r[2]]))
        self.assertEqual(y[0], 0)
        self.assertGreater(y[1], x[1])
        self.assertLess(y[2], x[2])

    def test_training_representation_separates_variable_state(self):
        a = sparse.csr_matrix([[0., 0., 1.], [1., 0., 1.], [2., 0., 1.]])
        b = sparse.csr_matrix([[3., 0., 1.], [4., 0., 1.], [5., 0., 1.]])
        features, _ = select_features([a, b], 2)
        self.assertIn(0, features)
        z, model = latent_space([a, b], features, 1)
        self.assertEqual(z[0].shape, (3, 1))
        self.assertGreater(np.linalg.norm(z[0].mean(0)-z[1].mean(0)), 1)
        self.assertTrue(np.isfinite(model['components']).all())


if __name__ == '__main__': unittest.main()
