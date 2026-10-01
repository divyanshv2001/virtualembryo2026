"""Bounded feasibility projection; convergence is checked, never assumed."""
import numpy as np


def project_mean_mass(candidate, reference, iterations=200, mean_tolerance=1e-5, mass_tolerance=1e-5):
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape or candidate.ndim != 2:
        raise ValueError('Shape mismatch')
    if not (np.isfinite(candidate).all() and np.isfinite(reference).all()) or min(candidate.min(),reference.min()) < 0:
        raise ValueError('Invalid nonnegative log-expression input')
    mask = candidate > 0
    target_mean = reference.mean(0)
    target_mass = np.expm1(reference).sum(1)
    impossible_columns = ((~mask.any(0)) & (target_mean > 0)) | (mask.any(0) & (target_mean == 0))
    impossible_rows = ((~mask.any(1)) & (target_mass > 0)) | (mask.any(1) & (target_mass == 0))
    if impossible_columns.any() or impossible_rows.any():
        return candidate.copy(), {'valid':False,'reason':'Locked support conflicts with a required nonzero/zero margin',
                 'impossible_columns':int(impossible_columns.sum()),'impossible_rows':int(impossible_rows.sum()),'iterations':0}
    value = candidate.copy()
    history = []
    for step in range(iterations + 1):
        mean_error = float(np.max(np.abs(value.mean(0)-target_mean)))
        mass_error = float(np.max(np.abs(np.expm1(value).sum(1)-target_mass)/np.maximum(target_mass,1e-12)))
        if step in [0,1,10,50,100,iterations] or (mean_error <= mean_tolerance and mass_error <= mass_tolerance):
            history.append({'step':step,'max_gene_log_mean_error':mean_error,'max_row_raw_mass_relative_error':mass_error})
        if mean_error <= mean_tolerance and mass_error <= mass_tolerance:
            return value, {'valid':True,'iterations':step,'history':history,'support_preserved':bool(np.array_equal(value>0,mask))}
        if step == iterations: break
        current = value.mean(0)
        value *= np.divide(target_mean,current,out=np.ones_like(current),where=current>0)[None,:]
        raw = np.expm1(value); mass = raw.sum(1)
        ratio = np.divide(target_mass,mass,out=np.ones_like(mass),where=mass>0)
        value = np.log1p(raw*ratio[:,None])
        if not np.isfinite(value).all():
            return candidate.copy(), {'valid':False,'reason':'Nonfinite iteration','iterations':step+1,'history':history}
    return value, {'valid':False,'reason':'Iteration budget exhausted; feasibility/convergence unestablished',
                   'iterations':iterations,'history':history,'support_preserved':bool(np.array_equal(value>0,mask))}
