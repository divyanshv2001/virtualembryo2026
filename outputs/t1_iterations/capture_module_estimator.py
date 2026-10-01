"""Fixed rank-two capture-level gene coexpression projection; no outcome fitting."""
import numpy as np


def fit(means,stages,cutoff,rank=2):
    means=np.asarray(means,float);stages=np.asarray(stages,float)
    levels=np.unique(stages)
    if means.ndim!=2 or len(means)!=len(stages) or len(levels)!=3 or not np.isfinite(means).all():
        raise ValueError('Invalid three-stage capture means')
    weights=np.array([1./(3*np.sum(stages==t)) for t in stages])
    center=weights@means;time=(stages-cutoff)/.25;time-=weights@time
    denominator=float(weights@(time*time))
    _,singular,basis=np.linalg.svd(np.sqrt(weights[:,None])*(means-center),full_matrices=False)
    tolerance=max(means.shape)*np.finfo(float).eps*(singular[0] if len(singular) else 0.)
    numerical_rank=int((singular>tolerance).sum())
    if numerical_rank<rank or denominator<=0:raise ValueError('Unsupported numerical rank or time contrast')
    ols=(weights*time)@means/denominator;basis=basis[:rank]
    projected=(ols@basis.T)@basis
    return projected,ols,{'numerical_rank':numerical_rank,'singular_values':singular.tolist(),
                          'weights':weights.tolist(),'projected_energy':float(projected@projected),
                          'unprojected_energy':float(ols@ols),'time_denominator':denominator}


def controls():
    stages=np.repeat([7.5,7.75,8.],3);time=(stages-8)/.25
    nuisance=np.tile([-1.,0.,1.],3)
    means=5+np.column_stack([time+nuisance,2*time-nuisance,.5*time+.2*nuisance])
    p,o,d=fit(means,stages,8.)
    np.testing.assert_allclose(p,o,atol=1e-12)
    for t in np.unique(stages):assert abs(sum(w for w,s in zip(d['weights'],stages) if s==t)-1/3)<1e-12
    keep=np.arange(len(stages))!=0;p,o,d=fit(means[keep],stages[keep],8.)
    np.testing.assert_allclose(p,o,atol=1e-12)
    try:fit(np.ones((9,3)),stages,8.)
    except ValueError:pass
    else:raise AssertionError('Rank-deficient identity must remain unsupported')


if __name__=='__main__':
    controls();print('Known rank-two slope, balanced weights, omission refit and deficient-identity controls passed')
