import unittest
import numpy as np
from iterate import change

class TransformChecks(unittest.TestCase):
    def test_zero_strength_is_identity(self):
        x=np.array([[0,1],[2,0]],dtype=np.float32)
        y,n=change(x,[-3,2],0)
        np.testing.assert_array_equal(x,y);self.assertEqual(n,0)
    def test_preservation_and_clipping_are_distinct(self):
        x=np.array([[0,1],[2,0]],dtype=np.float32)
        additive,n=change(x,[4,-8],.25)
        preserved,m=change(x,[4,-8],.25,True)
        np.testing.assert_array_equal(additive,[[1,0],[3,0]])
        np.testing.assert_array_equal(preserved,[[0,0],[3,0]])
        self.assertEqual(n,2);self.assertEqual(m,1)
    def test_nonfinite_changes_fail(self):
        with self.assertRaises(ValueError):change([[1,2]],[float('nan'),0],1)
    def test_inputs_unchanged(self):
        x=np.array([[0,1]],dtype=np.float32);copy=x.copy()
        change(x,[2,-2],.5,True)
        np.testing.assert_array_equal(x,copy)

if __name__=='__main__':unittest.main()
