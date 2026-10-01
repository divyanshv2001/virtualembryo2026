"""Projection integration controls: protected regions and strict rejection."""
import numpy as np
from cm_mean_mass_projection_fullpanel import project
from test_joint_margin_solver import main as solver_controls


def main():
    solver_controls(adaptive=True)
    reference = np.array([[1.,2.,9.],[2.,1.,8.],[3.,4.,7.]],np.float32)
    mask = np.array([True,True,False]); mapped = np.array([0,1]); guard = mapped
    result,audit = project(reference,reference,reference,mask,mapped,guard)
    assert audit['valid'] and np.array_equal(result,reference)
    impossible = reference.copy(); impossible[:2,0] = 0
    result,audit = project(impossible,reference,reference,mask,mapped,guard)
    assert result is None and not audit['valid']
    # A feasible margin match must still fail the existing covariance guard.
    donors = reference.copy(); donors[:,:2] *= .01
    result,audit = project(reference,reference,donors,mask,mapped,guard)
    assert result is None and not audit['post_cast_checks']['covariance_guard_passed']
    print('Projection integration controls passed: identity/protection, impossible support, strict covariance rejection.')


if __name__=='__main__': main()
