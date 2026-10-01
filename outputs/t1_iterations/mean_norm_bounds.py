"""Necessary bounds for donor-zero-locked mean and displacement matching."""
import numpy as np


def bounds(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape or a.ndim!=2 or len(a)==0 or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Invalid paired matrices')
    n=len(a);support=(a>0).sum(0);delta=(b-a).mean(0);unsupported=(support==0)&(delta!=0)
    minimum=np.divide((n*delta)**2,support,out=np.zeros_like(delta),where=support>0).sum()
    energy=float(((b-a)**2).sum());mean_energy=float((delta**2).sum())
    return {'unsupported_nonzero_mean_genes_exact':int(unsupported.sum()),'unsupported_nonzero_mean_genes_above_1e8':int(((support==0)&(np.abs(delta)>1e-8)).sum()),
            'unavoidable_mean_error_squared':float((delta[unsupported]**2).sum()),'desired_mean_shift_squared':mean_energy,
            'supported_coordinates_minimum_displacement_squared':float(minimum),'actual_displacement_squared':energy,
            'supported_minimum_vs_actual_energy':float(minimum/energy) if energy else None,
            'exact_support_and_energy_matching_necessary_conditions_pass':bool(not unsupported.any() and minimum<=energy+1e-8*max(energy,1.)),
            'sufficient_conditions_checked':False}


def controls():
    a=np.array([[0.,1.],[0.,1.]])
    assert bounds(a,a)['exact_support_and_energy_matching_necessary_conditions_pass']
    assert bounds(a,a+1)['unsupported_nonzero_mean_genes_exact']==1
    b=a.copy();b[:,1]+=2
    r=bounds(a,b);assert r['supported_coordinates_minimum_displacement_squared']==r['actual_displacement_squared']==8
    # Target spreads change over zeros; locking those zeros costs more at fixed mean.
    a=np.array([[1.],[0.]]);b=a+1;r=bounds(a,b)
    assert r['supported_minimum_vs_actual_energy']==2 and not r['exact_support_and_energy_matching_necessary_conditions_pass']
