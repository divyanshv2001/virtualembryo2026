"""Forecast-only displacement, ranks and regularized linear tangent diagnostics."""
import numpy as np
from scipy.stats import rankdata
from joint_margin_solver import joint_step


def alignment(proposed, surviving):
    a=float(np.linalg.norm(proposed)); b=float(np.linalg.norm(surviving))
    return {'proposed_l2':a,'surviving_l2':b,'survival_ratio':b/a if a else None,
            'cosine':float(np.sum(proposed*surviving)/(a*b)) if a and b else None}


def rank_changes(reference, value, chunk=512):
    changed_entries=changed_genes=positive_changed=0
    for start in range(0,reference.shape[1],chunk):
        sl=slice(start,start+chunk)
        changed=rankdata(reference[:,sl],axis=0,method='average')!=rankdata(value[:,sl],axis=0,method='average')
        changed_entries+=int(changed.sum());changed_genes+=int(changed.any(0).sum())
        positive_changed+=int((changed&(reference[:,sl]>0)).sum())
    return {'changed_entries':changed_entries,'entries':reference.size,'changed_genes':changed_genes,
            'genes':reference.shape[1],'changed_positive_entries':positive_changed,'positive_entries':int((reference>0).sum())}


def tangent(reference, direction, damping=1e-9):
    support=reference>0
    if np.any(direction[~support]!=0):raise ValueError('Tangent direction outside anchor support')
    ms=np.maximum(reference.mean(0),.1);rs=np.maximum(np.expm1(reference).sum(1),1e-12)
    def apply(v):return v.mean(0)/ms,(np.exp(reference)*v).sum(1)/rs
    column,row=apply(direction)
    removed=joint_step(reference,support,column,row,ms,rs,damping)
    result=direction+removed
    after_column,after_row=apply(result)
    before=float(np.linalg.norm(np.r_[column,row]));after=float(np.linalg.norm(np.r_[after_column,after_row]))
    return result,{'damping':damping,**alignment(direction,result),'linear_constraint_l2_before':before,
                   'linear_constraint_l2_after':after,'linear_constraint_ratio':after/before if before else None,
                   'finite':bool(np.isfinite(result).all()),'largest_dense_square_dimension':len(reference),
                   'interpretation':'Regularized linear tangent only; no nonlinear feasibility or benchmark score'}


class PastReadGuard:
    """Deny access to expression rows later than the frozen training cutoff."""
    def __init__(self,array,stages,cutoff):
        self.array=array;self.stages=stages;self.cutoff=cutoff;self.shape=array.shape;self.dtype=array.dtype
        self.reads=0;self.max_stage=None
    def __len__(self):return len(self.array)
    def __getitem__(self,key):
        rows=key[0] if isinstance(key,tuple) else key
        if isinstance(rows,slice):rows=np.arange(len(self.array))[rows]
        rows=np.asarray(rows)
        if rows.dtype==bool:rows=np.flatnonzero(rows)
        if rows.size and (not np.isfinite(self.stages[rows]).all() or np.any(self.stages[rows]>self.cutoff)):
            raise ValueError('Future expression read blocked')
        if rows.size:
            self.max_stage=max(float(self.stages[rows].max()),self.max_stage or -np.inf)
        self.reads+=1
        return self.array[key]
    def __array__(self,*args,**kwargs):raise ValueError('Unbounded expression materialization blocked')
