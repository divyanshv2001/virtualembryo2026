"""Bounded abundance-strength ablation on frozen conditional heads."""
import numpy as np
from conditional_positive_head import FullConditionalPositiveForecast


class StrengthTransportForecast(FullConditionalPositiveForecast):
    def predict(self,target,mode,strength=1.,seed=20260928,sampling='independent'):
        if mode!='abundance' or strength not in [.25,.5,1.,2.]:
            raise ValueError('Only predeclared abundance strengths are allowed')
        original=self.positive_coef
        # Scaling before the common cap and guard preserves the existing head
        # equation. Restore the fitted coefficient even if projection fails.
        self.positive_coef=original*strength
        try:
            result,indices,audit=super().predict(target,mode,1.,seed,sampling)
        finally:
            self.positive_coef=original
        audit.update(strength=strength,head_multiplier=strength,
            scope_strength='Change only conditional abundance displacement; fixed latent endpoint, caps, covariance guard, donors and scorer. Not faster biological time.')
        return result,indices,audit
