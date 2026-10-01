"""Numerical controls before the observation-only solver audit."""
import numpy as np
from joint_margin_solver import solve, residuals, joint_step


def main(adaptive=False):
    reference=np.array([[.4,1.2,0.],[1.1,.7,0.],[.6,.9,0.]])
    support=reference>0;n,g=reference.shape
    mean=reference.mean(0);mass=np.expm1(reference).sum(1)
    ms=np.maximum(mean,.1);rs=np.maximum(mass,1e-12)
    point=reference*.8;column,row,_=residuals(point,mean,mass,ms,rs)
    # Tiny dense Jacobian exists in this control only, never in production.
    locations=np.argwhere(support);jacobian=np.zeros((g+n,len(locations)))
    numerical=np.zeros_like(jacobian);epsilon=1e-6
    for k,(i,j) in enumerate(locations):
        jacobian[j,k]=1/(n*ms[j]);jacobian[g+i,k]=np.exp(point[i,j])/rs[i]
        plus=point.copy();minus=point.copy();plus[i,j]+=epsilon;minus[i,j]-=epsilon
        cp,rp,_=residuals(plus,mean,mass,ms,rs);cm,rm,_=residuals(minus,mean,mass,ms,rs)
        numerical[:,k]=(np.r_[cp,rp]-np.r_[cm,rm])/(2*epsilon)
    np.testing.assert_allclose(jacobian,numerical,rtol=1e-6,atol=1e-8)
    damping=.001
    expected=-jacobian.T@np.linalg.solve(jacobian@jacobian.T+damping*np.eye(g+n),np.r_[column,row])
    actual=joint_step(point,support,column,row,ms,rs,damping)
    np.testing.assert_allclose(actual[support],expected,rtol=1e-7,atol=1e-9)
    result,audit=solve(reference,reference,adaptive=adaptive)
    assert audit['valid'] and audit['iterations']==0
    result,audit=solve(point,reference,adaptive=adaptive)
    assert audit['valid'],audit
    np.testing.assert_allclose(result.mean(0),mean,atol=1e-5)
    np.testing.assert_allclose(np.expm1(result).sum(1),mass,rtol=1e-5)
    assert np.array_equal(result>0,support)
    _,audit=solve(np.array([[1.],[0.]]),np.ones((2,1)),adaptive=adaptive)
    assert not audit['valid'] and 'contradiction' in audit['reason']
    _,audit=solve(point,reference,iterations=0,adaptive=adaptive)
    assert not audit['valid']
    print('Joint solver controls passed: Jacobian finite differences, Schur/dense equivalence, identity, feasible perturbation, support contradiction, zero-budget failure.')


if __name__=='__main__':main()
