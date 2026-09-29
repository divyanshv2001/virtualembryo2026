"""Anchor slope adaptation with source levels held at anchor centers."""
import numpy as np
import torch
from feature_panel_forecast import FeaturePanelForecast
from ridge_conditional_head import conditional_ridge


class AnchorSlopeCalibration(FeaturePanelForecast):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        with torch.no_grad():
            z=self.net.encode(torch.tensor((self.donors[:,self.features]-self.center)/self.scale))[0]
        h=z.numpy().astype(float)-self.zcenter
        d=h.shape[1];size=len(self.mapped);self.anchor_latent_mean=h.mean(0)
        centered=h-self.anchor_latent_mean
        gram=centered.T@centered/len(h)+np.eye(d)
        self.anchor_positive_delta=np.zeros((d,size));self.anchor_detection_delta=np.zeros((d,size))
        self.anchor_positive_center=np.zeros((d,size));self.anchor_count=np.zeros(size,dtype=int)
        self.calibration_support=np.zeros(size,dtype=bool)
        for start in range(0,size,512):
            sl=slice(start,min(start+512,size));genes=self.mapped[sl]
            raw=np.expm1(self.donors[:,genes].astype(float));positive=(raw>0).astype(float);count=positive.sum(0)
            supported=(count>=20)&self.support[sl]
            logvalues=np.log(np.maximum(raw,1e-8))*positive
            positive_coef=conditional_ridge(h,positive,logvalues,ridge=1.)
            detection_coef=np.linalg.solve(gram,centered.T@(positive-positive.mean(0))/len(h))
            self.anchor_positive_delta[:,sl]=(positive_coef-self.positive_coef[:,sl])*supported
            self.anchor_detection_delta[:,sl]=(detection_coef-self.detection[:,sl])*supported
            self.anchor_positive_center[:,sl]=h.T@positive/np.maximum(count,1)
            self.anchor_count[sl]=count;self.calibration_support[sl]=supported
        self.audit.update(anchor_calibration_rows=len(h),anchor_supported_genes=int(self.calibration_support.sum()),
                          anchor_slope_ridge=1.,method='Past-anchor slope adaptation, source levels held at anchor centers',
                          limitation='Cross-sectional associations are not temporal velocities; only 1500 past anchors.')
        self.configure(0.,0.)
    def configure(self,positive_alpha,detection_alpha):
        if positive_alpha not in [0.,.5,1.] or detection_alpha not in [0.,.5,1.]:
            raise ValueError('Undeclared anchor strength')
        self.positive_alpha=positive_alpha;self.detection_alpha=detection_alpha
        self.audit.update(anchor_positive_slope_strength=positive_alpha,anchor_detection_slope_strength=detection_alpha)
    def positive_log_change(self,h0,h1,sl,target):
        return super().positive_log_change(h0,h1,sl,target)+self.positive_alpha*((h1-h0)@self.anchor_positive_delta[:,sl])
    def positive_log_value(self,h1,sl,target):
        delta=self.anchor_positive_delta[:,sl]
        correction=h1@delta-(self.anchor_positive_center[:,sl]*delta).sum(0)
        return super().positive_log_value(h1,sl,target)+self.positive_alpha*correction
    def detection_probability(self,h,sl):
        correction=(h-self.anchor_latent_mean)@self.anchor_detection_delta[:,sl]
        return np.clip(self.pmean[sl]+h@self.detection[:,sl]+self.detection_alpha*correction,1e-4,1-1e-4)
