"""Frozen within-support shuffled tangent control and shared positive scaling."""
import numpy as np
from projection_survival_diagnostics import tangent


def construct(reference, initial, seed=20260928):
    direction=initial.astype(float)-reference.astype(float)
    learned,learned_audit=tangent(reference.astype(float),direction)
    rng=np.random.default_rng(seed);shuffled=direction.copy();support=reference>0
    for gene in range(reference.shape[1]):
        rows=np.flatnonzero(support[:,gene]);shuffled[rows,gene]=rng.permutation(shuffled[rows,gene])
    null,null_audit=tangent(reference.astype(float),shuffled)
    ln=float(np.linalg.norm(learned));nn=float(np.linalg.norm(null))
    if not np.isfinite([ln,nn]).all() or min(ln,nn)<=0:raise ValueError('Zero/nonfinite tangent direction')
    null*=ln/nn
    bounds=[]
    for d in [learned,null]:
        negative=d<0;bounds.append(float(np.min(reference[negative]/(-d[negative]))) if negative.any() else float('inf'))
    scale=min(.25,.9*min(bounds))
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid common positivity scale')
    predictions={name:(reference.astype(float)+scale*d).astype(np.float32) for name,d in [('learned',learned),('shuffle',null)]}
    for value in predictions.values():
        if not np.isfinite(value).all() or not np.array_equal(value>0,support):raise ValueError('Cast/support violation')
    relative=float(abs(np.linalg.norm(null)-ln)/ln)
    if relative>1e-6:raise ValueError('Tangent norm matching failed')
    return predictions,{'shared_scale':scale,'learned_tangent_norm':ln,'shuffled_tangent_norm':float(np.linalg.norm(null)),
                        'relative_norm_match_error':relative,'learned_linear_audit':learned_audit,'shuffle_linear_audit':null_audit,'seed':seed}
