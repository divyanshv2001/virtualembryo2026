"""Damped Gauss-Newton with a cell-sized Schur complement, no dense variable Jacobian."""
import numpy as np
from support_mass_bounds import necessary_bounds


def residuals(value, target_mean, target_mass, mean_scale, mass_scale):
    with np.errstate(over='ignore', invalid='ignore'):
        raw=np.expm1(value)
        column=(value.mean(0)-target_mean)/mean_scale
        row=(raw.sum(1)-target_mass)/mass_scale
        objective=float(column@column+row@row)
    return column,row,objective


def joint_step(value, support, column, row, mean_scale, mass_scale, damping):
    n=len(value);a=1./(n*mean_scale)
    q=np.exp(value)*support/mass_scale[:,None]
    diagonal=support.sum(0)*a*a+damping
    cross=(q*a[None,:]).T
    row_diagonal=(q*q).sum(1)+damping
    schur=np.diag(row_diagonal)-cross.T@(cross/diagonal[:,None])
    row_weight=np.linalg.solve(schur,-row+cross.T@(column/diagonal))
    column_weight=(-column-cross@row_weight)/diagonal
    return (a[None,:]*column_weight[None,:]+q*row_weight[:,None])*support


def solve(candidate, reference, iterations=100, mean_tolerance=1e-5, mass_tolerance=1e-5,
          damping=.001, backoffs=(1.,.5,.25,.125,.0625,.03125), positivity_floor=1e-10):
    candidate=np.asarray(candidate,float);reference=np.asarray(reference,float)
    bounds=necessary_bounds(candidate,reference)
    support=candidate>0
    target_mean=reference.mean(0);target_mass=np.expm1(reference).sum(1)
    if bounds['bounds_rejected'] or np.any(support.any(1)&(target_mass==0)):
        return candidate.copy(),{'valid':False,'reason':'Necessary support/margin contradiction','bounds':bounds,'iterations':0}
    mean_scale=np.maximum(target_mean,.1);mass_scale=np.maximum(target_mass,1e-12)
    value=candidate.copy();history=[];reason='Iteration budget exhausted; feasibility unresolved'
    for step in range(iterations+1):
        column,row,objective=residuals(value,target_mean,target_mass,mean_scale,mass_scale)
        mean_error=float(np.max(np.abs(value.mean(0)-target_mean)))
        with np.errstate(over='ignore'):
            mass_error=float(np.max(np.abs(np.expm1(value).sum(1)-target_mass)/mass_scale))
        if step in [0,1,10,25,50,75,iterations] or (mean_error<=mean_tolerance and mass_error<=mass_tolerance):
            history.append({'step':step,'objective':objective,'max_gene_log_mean_error':mean_error,
                            'max_row_raw_mass_relative_error':mass_error})
        if np.isfinite(objective) and mean_error<=mean_tolerance and mass_error<=mass_tolerance:
            return value,{'valid':True,'iterations':step,'history':history,'support_preserved':bool(np.array_equal(value>0,support)),
                          'solver':'Structured damped Gauss-Newton','largest_dense_square_dimension':len(value)}
        if step==iterations:break
        if not np.isfinite(objective):reason='Nonfinite residual';break
        try:
            delta=joint_step(value,support,column,row,mean_scale,mass_scale,damping)
        except np.linalg.LinAlgError:
            reason='Cell Schur solve failed';break
        if not np.isfinite(delta).all():reason='Nonfinite step';break
        accepted=False
        for factor in backoffs:
            proposed=np.where(support,np.maximum(value+factor*delta,positivity_floor),0.)
            _,_,next_objective=residuals(proposed,target_mean,target_mass,mean_scale,mass_scale)
            if np.isfinite(next_objective) and next_objective<objective:
                value=proposed;accepted=True;break
        if not accepted:reason='No finite decreasing positivity-preserving step';break
    return value,{'valid':False,'reason':reason,'iterations':step,'history':history,'support_preserved':bool(np.array_equal(value>0,support)),
                  'solver':'Structured damped Gauss-Newton','largest_dense_square_dimension':len(value)}
