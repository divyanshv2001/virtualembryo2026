import unittest
import numpy as np
from transfer_genes import apply_unique_factors


class TransferTests(unittest.TestCase):
    def test_missing_and_ambiguous_genes_remain_exact_with_mass_conservation(self):
        x = np.log1p(np.array([[0., 100., 400., 9500.], [20., 80., 200., 9700.]])).astype(np.float32)
        result, audit = apply_unique_factors(x, ['A', 'B', 'duplicate', 'missing'],
            ['A', 'B', 'duplicate', 'duplicate'], np.array([2., .5, 9., 7.]))
        np.testing.assert_array_equal(result[:, 2:], x[:, 2:])
        np.testing.assert_array_equal(result == 0, x == 0)
        np.testing.assert_allclose(np.expm1(result).sum(1), np.expm1(x).sum(1), rtol=1e-6)
        self.assertEqual(audit['mapped_genes'], 2)
        self.assertFalse(np.array_equal(result[:, :2], x[:, :2]))

    def test_bad_factors_and_duplicate_official_panel_are_rejected(self):
        x = np.ones((2, 2), dtype=np.float32)
        with self.assertRaises(ValueError): apply_unique_factors(x, ['A', 'B'], ['A', 'B'], [0., 1.])
        with self.assertRaises(ValueError): apply_unique_factors(x, ['A', 'A'], ['A', 'B'], [1., 1.])


if __name__ == '__main__': unittest.main()
