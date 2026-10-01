"""Necessary conditions only: passing does not prove joint feasibility."""
import numpy as np


def necessary_bounds(candidate, reference, tolerance=1e-8):
    candidate=np.asarray(candidate,float);reference=np.asarray(reference,float)
    if candidate.shape!=reference.shape or candidate.ndim!=2 or not np.isfinite(candidate).all() or not np.isfinite(reference).all() or min(candidate.min(),reference.min())<0:
        raise ValueError('Invalid matched log-expression matrices')
    support=candidate>0;n=len(reference);m=support.sum(0)
    required_log_sum=reference.sum(0);mass=np.expm1(reference).sum(1)
    impossible_zero=bool(np.any((m==0)&(required_log_sum>0)) or np.any((m>0)&(required_log_sum==0)))
    average=np.divide(required_log_sum,m,out=np.zeros_like(required_log_sum),where=m>0)
    with np.errstate(over='ignore'):
        minimum_total=float(np.sum(m*np.expm1(average)))
    available=float(mass.sum())
    log_capacity=(support*np.log1p(mass)[:,None]).sum(0)
    excess=float(np.max(required_log_sum-log_capacity))
    mass_rejected=bool(minimum_total>available+tolerance*max(available,1.))
    capacity_rejected=bool(excess>tolerance)
    rejected=impossible_zero or mass_rejected or capacity_rejected
    return {'bounds_rejected':rejected,'valid':not rejected,
            'interpretation':'Proved infeasible for these locked margins/support' if rejected else 'Unresolved: necessary bounds pass, feasibility not established',
            'zero_support_conflict':impossible_zero,'minimum_raw_mass_sum':minimum_total if np.isfinite(minimum_total) else None,
            'minimum_raw_mass_overflow':bool(not np.isfinite(minimum_total)),'available_raw_mass_sum':available,
            'minimum_to_available_ratio':minimum_total/max(available,1e-12) if np.isfinite(minimum_total) else None,
            'raw_mass_bound_rejected':mass_rejected,'max_column_log_capacity_excess':excess,
            'column_capacity_bound_rejected':capacity_rejected,'tolerance':tolerance}
