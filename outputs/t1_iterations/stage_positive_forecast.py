"""Past-only stage-conditioned conditional positive expression regression."""
from collections import Counter
import numpy as np
import torch
from feature_panel_forecast import FeaturePanelForecast
from ridge_conditional_head import conditional_ridge


class StagePositiveForecast(FeaturePanelForecast):
    def __init__(self,*args,design='additive',**kwargs):
        if design not in ['additive','interaction']:raise ValueError('Undeclared stage design')
        super().__init__(*args,**kwargs)
        x,stages,cutoff=args[:3];panel,symbols=args[4:6]
        rows=np.flatnonzero(stages<=cutoff)
        self.stage_design=design
        self.stage_mean=float(np.mean(stages[rows]));self.stage_scale=float(np.std(stages[rows]))
        if self.stage_scale<=0:raise ValueError('Multiple observed stages required')
        counts=Counter(symbols);lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
        feat=[lookup[panel[i]] for i in self.features]
        atlas=[lookup[panel[i]] for i in self.mapped]
        with torch.no_grad():
            h=self.net.encode(torch.tensor((np.asarray(x[np.ix_(rows,feat)])-self.center)/self.scale))[0].numpy().astype(float)-self.zcenter
        augmented=self.design_matrix(h,stages[rows])
        d=augmented.shape[1];g=len(self.mapped)
        self.stage_coef=np.zeros((d,g));self.stage_centroid=np.zeros((d,g));self.stage_response_mean=np.zeros(g)
        for start in range(0,g,512):
            sl=slice(start,min(start+512,g))
            raw=np.asarray(x[np.ix_(rows,np.asarray(atlas)[sl])],float)
            detected=(raw>0).astype(float);den=np.maximum(detected.sum(0),1.)
            response=np.log(np.maximum(np.expm1(raw),1e-8))*detected
            self.stage_coef[:,sl]=conditional_ridge(augmented,detected,response,ridge=1.)
            self.stage_centroid[:,sl]=augmented.T@detected/den
            self.stage_response_mean[sl]=response.sum(0)/den
        self.audit.update(positive_stage_design=design,stage_mean=self.stage_mean,stage_scale=self.stage_scale,
                          stage_fit_rows=len(rows),stage_fit_max=float(stages[rows].max()),stage_ridge=1.,
                          scope='Own stage-conditioned positive regression hypothesis; latent dynamics and global detection unchanged. Past-only normalization, conditional support20, original output constraints.')

    def design_matrix(self,h,stage):
        tau=np.broadcast_to((np.asarray(stage)-self.stage_mean)/self.stage_scale,(len(h),))[:,None]
        return np.concatenate([h,tau]+([h*tau] if self.stage_design=='interaction' else []),axis=1)

    def positive_log_change(self,h0,h1,sl,target):
        return (self.design_matrix(h1,target)-self.design_matrix(h0,self.cutoff))@self.stage_coef[:,sl]

    def positive_log_value(self,h1,sl,target):
        return self.stage_response_mean[sl]+self.design_matrix(h1,target)@self.stage_coef[:,sl]-(self.stage_centroid[:,sl]*self.stage_coef[:,sl]).sum(0)
