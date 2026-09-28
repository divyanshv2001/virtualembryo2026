import unittest
import numpy as np
from cell_mixture import mix_cells


class MixtureTests(unittest.TestCase):
    def test_complete_cells_and_provenance_survive_integration(self):
        a = np.arange(60,dtype=np.float32).reshape(12,5)
        b = a+100
        result, origin = mix_cells(a,b,.25,33)
        self.assertEqual(int((origin[:,0] == 0).sum()),3)
        for row,(source,index) in zip(result,origin): np.testing.assert_array_equal(row,[a,b][source][index])
        np.testing.assert_array_equal(result,mix_cells(a,b,.25,33)[0])


if __name__ == '__main__': unittest.main()
