"""Quarter-amplitude witness initialization and strict projection checks."""
import numpy as np
from cm_anchor_support_shrinkage_fullpanel import initialize, project


def main():
    anchor = np.array([[1.,0.,9.],[2.,1.,8.],[3.,4.,7.]],np.float32)
    corrected = np.array([[0.,3.,9.],[3.,2.,8.],[3.,4.,7.]],np.float32)
    mask = np.array([True,True,False]); mapped = np.array([0,1])
    initial = initialize(corrected,anchor,mask,mapped)
    assert np.array_equal(initial[:2,:2]>0,anchor[:2,:2]>0)
    assert initial[0,0]==1 and initial[0,1]==0
    assert initial[1,0]==2.25 and initial[1,1]==1.25
    assert np.array_equal(initial[~mask],corrected[~mask])
    assert np.array_equal(initial[:,2],corrected[:,2])
    assert np.array_equal(initialize(anchor,anchor,mask,mapped),anchor)
    result,audit = project(initial,anchor,anchor,mask,mapped,mapped)
    assert audit['valid'],audit
    np.testing.assert_allclose(result[:2,:2].astype(float).mean(0),anchor[:2,:2].astype(float).mean(0),atol=1e-5)
    np.testing.assert_allclose(np.expm1(result[:2,:2].astype(float)).sum(1),np.expm1(anchor[:2,:2].astype(float)).sum(1),rtol=1e-5)
    assert np.array_equal(result[:2,:2]>0,anchor[:2,:2]>0)
    print('Quarter amplitude checks passed: exact support/fill, fixed strength, protected regions, zero correction identity, feasible projection.')


if __name__=='__main__':main()
