"""Fixed observation-level odds offsets, shared by both temporal endpoints."""
import numpy as np
from scipy.special import expit, logit
from partial_anchor_forecast import PartialAnchorForecast


def observation_offset(source, observed):
    """Jeffreys regularization; inputs are current-stage expression only."""
    if min(len(source), len(observed)) < 20:
        raise ValueError('Insufficient current observation support')
    source_q = ((source > 0).sum(0) + .5) / (len(source) + 1.)
    observed_q = ((observed > 0).sum(0) + .5) / (len(observed) + 1.)
    return logit(observed_q) - logit(source_q)


def apply_offset(probability, offsets, rows, alpha):
    result = probability.copy()
    if alpha:
        result[rows] = expit(logit(result[rows]) + alpha * offsets)
    return np.clip(result, 1e-4, 1 - 1e-4)


class CMObservationForecast(PartialAnchorForecast):
    def set_observation_calibration(self, offsets, rows, alpha, scope):
        if alpha not in [0., .25, .5] or scope not in ['none', 'global', 'cm']:
            raise ValueError('Undeclared observation calibration')
        self.observation_offsets = np.asarray(offsets)
        self.observation_rows = np.asarray(rows, bool)
        self.observation_alpha = alpha
        self.audit.update(observation_scope=scope, observation_alpha=alpha,
                          calibration_endpoint_policy='Same current-only offset applied to p0/p1; calibrated switching denominators',
                          observation_calibrated_cells=int(self.observation_rows.sum()))

    def detection_probability(self, h, sl):
        probability = super().detection_probability(h, sl)
        return apply_offset(probability, self.observation_offsets[sl],
                            self.observation_rows, self.observation_alpha)
