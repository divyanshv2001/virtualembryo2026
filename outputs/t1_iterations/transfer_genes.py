"""Explicit gene-factor mapping for a future candidate that passes promotion."""
from collections import Counter
import numpy as np


def apply_unique_factors(donors, official_symbols, atlas_symbols, factors):
    """Preserve unmapped/ambiguous genes exactly and conserve mapped library mass.

    factors is a gene vector or a donor-by-gene matrix in atlas order. This helper
    does not select a model, infer a future distribution, export or submit files.
    """
    donors = np.asarray(donors)
    if donors.dtype != np.float32: raise ValueError('Require float32 donors to preserve protected genes exactly')
    factors = np.asarray(factors, dtype=float)
    if donors.ndim != 2 or donors.shape[1] != len(official_symbols): raise ValueError('Donor panel mismatch')
    if len(set(official_symbols)) != len(official_symbols): raise ValueError('Duplicate official symbols')
    if factors.shape not in [(len(atlas_symbols),), (len(donors), len(atlas_symbols))]: raise ValueError('Factor shape mismatch')
    if not np.isfinite(donors).all() or (donors < 0).any(): raise ValueError('Invalid donors')
    if not np.isfinite(factors).all() or (factors <= 0).any(): raise ValueError('Invalid factors')
    counts = Counter(atlas_symbols)
    unique = {s:i for i, s in enumerate(atlas_symbols) if s and counts[s] == 1}
    mapped = [i for i, s in enumerate(official_symbols) if s in unique]
    atlas = [unique[official_symbols[i]] for i in mapped]
    result = donors.astype(np.float32, copy=True)
    if mapped:
        abundance = np.expm1(donors[:, mapped].astype(float))
        mass = abundance.sum(1)
        selected = factors[atlas] if factors.ndim == 1 else factors[:, atlas]
        updated = abundance*selected
        total = updated.sum(1)
        if not np.isfinite(updated).all(): raise ValueError('Factor application overflow')
        ratio = np.divide(mass, total, out=np.ones_like(mass), where=total > 0)
        result[:, mapped] = np.log1p(updated*ratio[:, None]).astype(np.float32)
    return result, {'mapped_genes':len(mapped), 'protected_genes':len(official_symbols)-len(mapped),
        'policy':'Unique exact symbols only; mapped abundance mass conserved within each donor. Protected genes unchanged. No full-panel renormalization.'}
