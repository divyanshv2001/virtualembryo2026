"""Recent-stage detection probabilities with unchanged global abundance heads."""
import numpy as np
from feature_panel_forecast import FeaturePanelForecast
from windowed_positive_head import PastRowsView


class RecentDetectionForecast(FeaturePanelForecast):
    def __init__(self,*args,head_stages=2,**kwargs):
        if head_stages not in [1,2,3]:raise ValueError('Undeclared detection window')
        super().__init__(*args,**kwargs)
        x,stages,cutoff=args[:3];past=np.flatnonzero(stages<=cutoff)
        selected=np.unique(stages[past])[-head_stages:];rows=past[np.isin(stages[past],selected)]
        recent=FeaturePanelForecast(PastRowsView(x,rows),stages[rows],cutoff,*args[3:],**kwargs)
        self.recent_detection=recent.detection;self.recent_pmean=recent.pmean;self.recent_center=recent.zcenter
        self.detection_support=recent.support
        self.audit.update(method='Frozen CNF, global full conditional abundance, recent-stage linear detection',recent_detection_stages=selected.tolist(),recent_detection_fit_cells=len(rows),scope='Only detection probabilities change; global positive coefficients/centroids and encoder/flow fixed. Past-only local stationarity hypothesis, not published biological-model reproduction.')

    def detection_probability(self,h,sl):
        # Forecast passes globally centered latent coordinates; recenter solely
        # for the recent detection head without changing abundance regression.
        probability=np.clip(self.recent_pmean[sl]+(h+self.zcenter-self.recent_center)@self.recent_detection[:,sl],1e-4,1-1e-4)
        # Below-support genes have zero recent slopes and constant probabilities.
        return probability
