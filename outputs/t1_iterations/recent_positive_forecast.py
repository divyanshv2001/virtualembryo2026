"""Recent-stage conditional abundance coefficients with global detection fixed."""
import numpy as np
from feature_panel_forecast import FeaturePanelForecast
from windowed_positive_head import PastRowsView


class RecentPositiveForecast(FeaturePanelForecast):
    def __init__(self,x,stages,cutoff,donors,panel,symbols,net,center,scale,features,guard_features,head_stages):
        if head_stages not in [1,2,3]:raise ValueError('Undeclared positive window')
        super().__init__(x,stages,cutoff,donors,panel,symbols,net,center,scale,features,guard_features)
        selected=np.unique(stages[stages<=cutoff])[-head_stages:]
        rows=np.flatnonzero(np.isin(stages,selected))
        recent=FeaturePanelForecast(PastRowsView(x,rows),stages[rows],cutoff,donors,panel,symbols,net,center,scale,features,guard_features)
        self.positive_coef=recent.positive_coef.copy()
        # Both centroids describe the same positive cells, in different latent
        # origins. Convert the recent centroid into the retained global frame.
        self.positive_center=recent.positive_center+recent.zcenter[:,None]-self.zcenter[:,None]
        self.positive_mean=recent.positive_mean.copy()
        self.audit.update(positive_fit_stages=selected.tolist(),positive_window=head_stages,
                          positive_fit_rows=len(rows),detection_policy='Unchanged all-past linear detection and support',
                          scope='Only positive conditional coefficients/intercepts change; frozen encoder/dynamics, global detection, constraints and scoring unchanged.')
