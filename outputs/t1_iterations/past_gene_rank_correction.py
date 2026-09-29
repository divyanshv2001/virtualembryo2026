"""Past-stage gene trend correction for a frozen full-panel forecast.

This is our bounded temporal-rank prior, not a DESeq2 or ashr reproduction.
"""
from collections import Counter
import numpy as np


def fit_consistent_gene_trends(x, stages, panel, symbols, donor, top_k):
    if top_k not in (64, 256):
        raise ValueError('Unplanned gene budget')
    stages = np.asarray(stages)
    windows = [np.flatnonzero(stages == s) for s in (8., 8.25, 8.5)]
    if min(map(len, windows)) < 100:
        raise ValueError('Insufficient permitted past stages')
    counts = Counter(symbols)
    lookup = {s:i for i,s in enumerate(symbols) if s and counts[s] == 1}
    mapped = np.array([i for i,s in enumerate(panel) if s in lookup], dtype=int)
    atlas = np.array([lookup[panel[i]] for i in mapped], dtype=int)
    means = np.empty((3, len(mapped)), dtype=float)
    for start in range(0, len(mapped), 256):
        sl = slice(start, min(start+256, len(mapped)))
        for t,rows in enumerate(windows):
            means[t, sl] = np.asarray(x[np.ix_(rows, atlas[sl])], dtype=np.float32).mean(0, dtype=np.float64)
    first, second = means[1]-means[0], means[2]-means[1]
    donor_mean = donor[:,mapped].mean(0)
    eligible = (first*second > 0) & (donor_mean > .02) & ((donor[:,mapped] > 0).mean(0) >= .05)
    priority = np.minimum(abs(first), abs(second))
    priority[~eligible] = -np.inf
    selected_local = np.argsort(-priority, kind='stable')[:top_k]
    selected_local = selected_local[np.isfinite(priority[selected_local])]
    signed_rate = 2*(first[selected_local]+second[selected_local])  # 1-day extrapolation from quarter-day steps
    adjustment = np.clip(signed_rate, -.15, .15)
    return mapped[selected_local], adjustment, {
        'source_stages':[8.,8.25,8.5], 'stage_row_counts':[len(r) for r in windows],
        'mapped_genes':len(mapped),'eligible_genes':int(eligible.sum()),
        'selected_genes':len(selected_local),'top_k':top_k,
        'mean_abs_adjustment':float(np.mean(abs(adjustment))) if len(adjustment) else 0.,
        'max_abs_adjustment':float(np.max(abs(adjustment))) if len(adjustment) else 0.,
        'limitation':'Source atlas cross-sectional trend extrapolated 1 day; no independent embryo or challenge-target fit.'}


def apply_positive_only_correction(forecast, genes, adjustment, strength):
    if strength not in (.5, 1.):
        raise ValueError('Unplanned correction strength')
    if forecast.dtype != np.float32 or forecast.ndim != 2:
        raise ValueError('Full-panel float32 forecast required')
    if len(genes) != len(adjustment):
        raise ValueError('Gene/adjustment mismatch')
    prediction = forecast.copy()
    if len(genes):
        original = prediction[:,genes]
        changed = np.where(original > 0, np.maximum(0,original+strength*adjustment),0)
        prediction[:,genes] = changed.astype(np.float32)
    return prediction
