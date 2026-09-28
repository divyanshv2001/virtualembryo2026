"""Reproducible mixtures retain complete specialist cells and source provenance."""
import numpy as np


def mix_cells(first, second, first_weight, seed):
    if first.shape != second.shape or first.ndim != 2 or first.dtype != np.float32 or second.dtype != np.float32:
        raise ValueError('Require matching full-panel float32 matrices')
    if not np.isfinite(first_weight) or not 0 <= first_weight <= 1: raise ValueError('Invalid mixture weight')
    rng = np.random.default_rng(seed); n = len(first); count = int(np.floor(n*first_weight+.5))
    a = rng.permutation(n)[:count]; b = rng.permutation(n)[:n-count]
    provenance = np.concatenate([np.column_stack([np.zeros(len(a),dtype=int),a]),
        np.column_stack([np.ones(len(b),dtype=int),b])])
    order = rng.permutation(n)
    return np.concatenate([first[a],second[b]],axis=0)[order], provenance[order]
