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


def test_matched_mixture_preserves_each_donor_once_and_endpoints():
    from cell_mixture import mix_matched_cells
    first=np.arange(65,dtype=np.float32).reshape(13,5);second=first+100
    for weight in [0.,.25,.5,.75,1.]:
        result,origin=mix_matched_cells(first,second,weight,33)
        np.testing.assert_array_equal(np.sort(origin[:,1]),np.arange(13))
        assert (origin[:,0]==0).sum()==int(np.floor(13*weight+.5))
        for row,(source,index) in zip(result,origin):
            np.testing.assert_array_equal(row,[first,second][source][index])
        np.testing.assert_array_equal(result,mix_matched_cells(first,second,weight,33)[0])
    import pytest
    with pytest.raises(ValueError,match='weight'):
        mix_matched_cells(first,second,float('nan'),33)
