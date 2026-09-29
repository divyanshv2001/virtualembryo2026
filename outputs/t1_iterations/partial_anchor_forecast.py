"""Bounded partial anchor calibration of the existing log1p decoder."""
from log1p_positive_forecast import Log1pPositiveForecast


class PartialAnchorForecast(Log1pPositiveForecast):
    def configure(self, positive_alpha, detection_alpha):
        if positive_alpha not in [0., .25, .5, .75, 1.] or detection_alpha not in [0., .25, .5, .75, 1.]:
            raise ValueError('Undeclared partial anchor strength')
        self.positive_alpha = positive_alpha
        self.detection_alpha = detection_alpha
        self.audit.update(anchor_positive_slope_strength=positive_alpha,
                          anchor_detection_slope_strength=detection_alpha)
