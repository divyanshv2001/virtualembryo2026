import unittest
import numpy as np
from composition import allocate,forecast_weights

class CompositionChecks(unittest.TestCase):
    def test_identity_retains_empirical_proportions(self):
        types,w,m=forecast_weights(['A','B'],['A','A','B','C'],0)
        np.testing.assert_allclose(w,[.5,.25,.25]);np.testing.assert_allclose(m,1)
    def test_new_labels_not_assigned_infinite_growth(self):
        types,w,m=forecast_weights(['A']*99+['B'],['A','B','B','C'],1,2)
        self.assertTrue(np.isfinite(w).all());self.assertAlmostEqual(w.sum(),1)
        self.assertEqual(m[list(types).index('B')],2)
        self.assertEqual(m[list(types).index('C')],1)
    def test_allocation_preserves_exact_count(self):
        counts=allocate([.333,.333,.334],2000)
        self.assertEqual(counts.sum(),2000);self.assertTrue((counts>=0).all())
    def test_parameters_fail_closed(self):
        for strength,cap in ((-1,2),(2,2),(.5,.5),(float('nan'),2)):
            with self.assertRaises(ValueError):forecast_weights(['A'],['A'],strength,cap)

if __name__=='__main__':unittest.main()
