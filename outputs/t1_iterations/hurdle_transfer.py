"""Separate past-only detection and conditional-positive abundance dynamics."""
import numpy as np
from detection_transfer import DetectionTransfer
from robust_population import stable_slope


class HurdleTransfer(DetectionTransfer):
    def __init__(self, x, stages, cutoff, *args, **kwargs):
        super().__init__(x, stages, cutoff, *args, **kwargs)
        self.original_state_slope = self.model.state_slope.copy()
        self.positive_state_slope = np.zeros_like(self.original_state_slope)
        rows = np.flatnonzero(stages <= cutoff)
        recent = np.unique(stages[rows])[-3:]
        labels = self.model.clusterer.labels_
        for k, support in enumerate(self.model.state_support):
            if support < 8: continue
            groups = [rows[(stages[rows] == t) & (labels == k)] for t in recent]
            for start in range(0, x.shape[1], 512):
                means = []; errors = []; supports = []
                for group in groups:
                    values = np.expm1(np.asarray(x[group, start:start+512], dtype=float))
                    n = (values > 0).sum(0)
                    mean = values.sum(0)/np.maximum(n, 1)
                    variance = np.maximum((values**2).sum(0)/np.maximum(n, 1)-mean**2, 0)
                    means.append(np.log(mean+.1))
                    errors.append(np.sqrt(variance/np.maximum(n, 1))/(mean+.1))
                    supports.append(n)
                n = np.min(supports, axis=0)
                self.positive_state_slope[k, start:start+512] = stable_slope(
                    recent, np.stack(means), np.stack(errors))*(n >= 8)*n/(n+64)

    def predict_hurdle(self, target, detection=.5, expression=.25, cap=.02, positive=True):
        self.model.state_slope = self.positive_state_slope if positive else self.original_state_slope
        try: return self.predict_detection(target, detection, expression, cap)
        finally: self.model.state_slope = self.original_state_slope

    def save(self, path):
        super().save(path)
        with np.load(path) as source: fields = {k:source[k] for k in source.files}
        np.savez_compressed(path, **fields, positive_state_slope=self.positive_state_slope)
