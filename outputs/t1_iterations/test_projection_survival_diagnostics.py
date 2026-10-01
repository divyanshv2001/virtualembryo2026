import numpy as np
from projection_survival_diagnostics import alignment,rank_changes,tangent,PastReadGuard


def main():
    reference=np.array([[.4,1.2,0.],[1.1,.7,0.],[.6,.9,0.]])
    direction=np.array([[.2,-.1,0.],[-.1,.2,0.],[.1,-.2,0.]])
    projected,audit=tangent(reference,direction)
    support=reference>0;locations=np.argwhere(support);n,g=reference.shape
    ms=np.maximum(reference.mean(0),.1);rs=np.maximum(np.expm1(reference).sum(1),1e-12)
    jacobian=np.zeros((g+n,len(locations)))
    for k,(i,j) in enumerate(locations):jacobian[j,k]=1/(n*ms[j]);jacobian[g+i,k]=np.exp(reference[i,j])/rs[i]
    v=direction[support];expected=v-jacobian.T@np.linalg.solve(jacobian@jacobian.T+1e-9*np.eye(g+n),jacobian@v)
    np.testing.assert_allclose(projected[support],expected,rtol=1e-5,atol=1e-8)
    assert audit['linear_constraint_ratio']<1e-5 and np.all(projected[~support]==0)
    zero,a=tangent(reference,np.zeros_like(reference));assert np.array_equal(zero,np.zeros_like(zero)) and a['survival_ratio'] is None and a['cosine'] is None
    assert alignment(direction,direction)['cosine']>1-1e-12
    ranks=rank_changes(reference,reference);assert ranks['changed_entries']==0 and ranks['positive_entries']==6
    assert rank_changes(reference,reference[::-1])['changed_entries']>0
    guarded=PastReadGuard(reference,np.array([1.,2.,3.]),2.)
    np.testing.assert_array_equal(guarded[np.ix_([0,1],[0,1])],reference[:2,:2])
    for access in [lambda:guarded[2],lambda:np.asarray(guarded)]:
        try:access()
        except ValueError:pass
        else:raise AssertionError('Future/unbounded read accepted')
    print('Survival controls passed: tangent/dense equivalence, constraint reduction, zero identity/null ratios, tie-aware ranks, future-read guard.')


if __name__=='__main__':main()
