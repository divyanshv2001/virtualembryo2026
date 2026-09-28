"""Past-only detection trends with measured within-state abundance donors."""
from collections import Counter
import numpy as np
from challenge_transfer import ChallengeTransfer
from robust_population import stable_slope, covariance_change


class DetectionTransfer(ChallengeTransfer):
    def __init__(self, x, stages, cutoff, donors, official_symbols, atlas_symbols, **kwargs):
        super().__init__(x, stages, cutoff, donors, official_symbols, atlas_symbols, **kwargs)
        rows = np.flatnonzero(stages <= cutoff)
        recent = np.unique(stages[rows])[-3:]
        if len(recent) != 3: raise ValueError('Three past stages required')
        labels = self.model.clusterer.labels_
        counts = Counter(atlas_symbols)
        lookup = {s:i for i, s in enumerate(atlas_symbols) if s and counts[s] == 1}
        self.mapped = np.array([i for i,s in enumerate(official_symbols) if s in lookup])
        self.atlas_mapped = np.array([lookup[official_symbols[i]] for i in self.mapped])
        self.detection_slope = np.zeros((len(self.model.state_support), len(self.mapped)))
        for k, support in enumerate(self.model.state_support):
            if support < 8: continue
            groups = [rows[(stages[rows] == t) & (labels == k)] for t in recent]
            for start in range(0, len(self.mapped), 512):
                genes = self.atlas_mapped[start:start+512]
                probabilities = []; errors = []
                for group in groups:
                    detected = np.asarray(x[np.ix_(group, genes)]) > 0
                    p = (detected.sum(0)+1)/(len(group)+2)
                    probabilities.append(p)
                    errors.append(np.sqrt(p*(1-p)/len(group)))
                self.detection_slope[k, start:start+512] = stable_slope(
                    recent, np.stack(probabilities), np.stack(errors))*support/(support+64)

    def predict_detection(self, target, detection=.5, expression=.25, cap=.02):
        if not 0 <= detection <= 1 or not 0 <= cap <= .05:
            raise ValueError('Detection strength/cap outside frozen bounds')
        base, indices, base_audit = self.predict(target, 'state', .5, expression)
        if detection == 0: return base, indices, {**base_audit, 'detection_strength':0.}
        candidate = base.copy()
        labels = self.labels[indices]; trusted = self.trusted[indices]
        additions = deletions = 0
        # Same gene/state random stream across strengths, never searched on target.
        for k in range(len(self.detection_slope)):
            group = np.flatnonzero((labels == k) & trusted)
            if len(group) < 8: continue
            for j, gene in enumerate(self.mapped):
                change = np.clip((target-self.model.cutoff)*detection*self.detection_slope[k,j], -cap, cap)
                n = int(np.floor(abs(change)*len(group)+.5))
                if not n: continue
                rng = np.random.default_rng(np.random.SeedSequence([20260928, k, int(gene)]))
                values = base[group, gene]
                positive = group[values > 0]; zero = group[values == 0]
                if change > 0 and len(positive) and len(zero):
                    chosen = rng.permutation(zero)[:n]
                    candidate[chosen, gene] = base[rng.choice(positive, len(chosen), replace=True), gene]
                    additions += len(chosen)
                elif change < 0 and len(positive):
                    chosen = rng.permutation(positive)[:n]
                    candidate[chosen, gene] = 0
                    deletions += len(chosen)
        # Preserve protected genes bit-for-bit and mapped abundance mass per cell.
        old = np.expm1(base[:, self.mapped].astype(float)).sum(1)
        values = np.expm1(candidate[:, self.mapped].astype(float)); total = values.sum(1)
        empty = (total == 0) & (old > 0)
        values[empty] = np.expm1(base[np.ix_(empty, self.mapped)].astype(float))
        total = values.sum(1)
        ratio = np.divide(old, total, out=np.ones_like(old), where=total > 0)
        candidate[:, self.mapped] = np.log1p(values*ratio[:,None]).astype(np.float32)
        cov = covariance_change(self.donors[:, self.official_features], candidate[:, self.official_features])
        audit = {**base_audit, 'detection_strength':detection, 'maximum_probability_change':cap,
            'attempted_additions':additions, 'attempted_deletions':deletions,
            'detection_candidate_covariance_change':cov, 'detection_guard_rejected':cov > self.covariance_limit}
        if cov > self.covariance_limit: candidate = base
        audit['actual_additions'] = int(((base == 0) & (candidate > 0)).sum())
        audit['actual_deletions'] = int(((base > 0) & (candidate == 0)).sum())
        return candidate, indices, audit

    def save(self, path):
        super().save(path)
        with np.load(path) as source: fields = {k:source[k] for k in source.files}
        np.savez_compressed(path, **fields, mapped=self.mapped,
            atlas_mapped=self.atlas_mapped, detection_slope=self.detection_slope)
