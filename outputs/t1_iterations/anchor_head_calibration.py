"""Past-anchor decoder intercept calibration with frozen atlas slopes."""
import numpy as np
import torch
from feature_panel_forecast import FeaturePanelForecast


class AnchorHeadCalibration(FeaturePanelForecast):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        with torch.no_grad():
            z=self.net.encode(torch.tensor((self.donors[:,self.features]-self.center)/self.scale))[0]
        h=z.numpy().astype(float)-self.zcenter
        size=len(self.mapped)
        self.anchor_positive_offset=np.zeros(size);self.anchor_detection_offset=np.zeros(size)
        self.anchor_count=np.zeros(size,dtype=int);self.calibration_support=np.zeros(size,dtype=bool)
        for start in range(0,size,512):
            sl=slice(start,min(start+512,size));genes=self.mapped[sl]
            raw=np.expm1(self.donors[:,genes].astype(float));positive=raw>0;count=positive.sum(0)
            supported=(count>=20)&self.support[sl]
            predicted=FeaturePanelForecast.positive_log_value(self,h,sl,self.cutoff)
            residual=((np.log(np.maximum(raw,1e-8))-predicted)*positive).sum(0)/np.maximum(count,1)
            self.anchor_positive_offset[sl]=np.clip(residual,-np.log(2.),np.log(2.))*supported
            unbounded=self.pmean[sl]+h@self.detection[:,sl]
            self.anchor_detection_offset[sl]=np.clip(positive.mean(0)-unbounded.mean(0),-.25,.25)*supported
            self.anchor_count[sl]=count;self.calibration_support[sl]=supported
        self.audit.update(anchor_calibration_rows=len(self.donors),anchor_supported_genes=int(self.calibration_support.sum()),
                          positive_offset_cap=float(np.log(2.)),detection_intercept_offset_cap=.25,
                          method='Observed past-anchor intercept calibration; source slopes/dynamics unchanged',
                          limitation='Atlas-anchor differences may be biological as well as technical; not ComBat empirical Bayes.')
        self.configure(0.,0.)
    def configure(self,positive_alpha,detection_alpha):
        if positive_alpha not in [0.,.5,1.] or detection_alpha not in [0.,.5,1.]:raise ValueError('Undeclared anchor strength')
        self.positive_alpha=positive_alpha;self.detection_alpha=detection_alpha
        self.audit.update(anchor_positive_strength=positive_alpha,anchor_detection_strength=detection_alpha)
    def positive_log_value(self,h1,sl,target):
        return super().positive_log_value(h1,sl,target)+self.positive_alpha*self.anchor_positive_offset[sl]
    def detection_probability(self,h,sl):
        return np.clip(self.pmean[sl]+h@self.detection[:,sl]+self.detection_alpha*self.anchor_detection_offset[sl],1e-4,1-1e-4)
