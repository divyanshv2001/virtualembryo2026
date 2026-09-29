import unittest
import numpy as np
from past_gene_rank_correction import apply_positive_only_correction


class PastGeneRankCorrectionTest(unittest.TestCase):
    def test_preserves_zeros_and_unselected_genes(self):
        x = np.array([[0, .05, 1, 2],[1, 0, 1, 0]],dtype=np.float32)
        y = apply_positive_only_correction(x,np.array([0,1]),np.array([-.15,.1]),1.)
        np.testing.assert_array_equal(y[:,2:],x[:,2:])
        self.assertEqual(y[0,0],0)
        self.assertEqual(y[1,1],0)
        self.assertEqual(y[1,0],.85)
        self.assertAlmostEqual(float(y[0,1]),.15,places=6)
        np.testing.assert_array_equal(x[:,2:],y[:,2:])


if __name__=='__main__':unittest.main()
