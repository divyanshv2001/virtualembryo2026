"""Conditional positive decoder in scoring log1p units, with raw-space guards."""
from collections import Counter
import numpy as np
import torch
from anchor_slope_calibration import AnchorSlopeCalibration
from ridge_conditional_head import conditional_ridge


class Log1pPositiveForecast(AnchorSlopeCalibration):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        x,stages,cutoff=args[:3];panel,symbols=args[4:6]
        rows=np.flatnonzero(stages<=cutoff);counts=Counter(symbols)
        lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
        feat=np.array([lookup[panel[i]] for i in self.features]);atlas=np.array([lookup[panel[i]] for i in self.mapped])
        with torch.no_grad():
            h=self.net.encode(torch.tensor((np.asarray(x[np.ix_(rows,feat)])-self.center)/self.scale))[0].numpy().astype(float)-self.zcenter
            ah=self.net.encode(torch.tensor((self.donors[:,self.features]-self.center)/self.scale))[0].numpy().astype(float)-self.zcenter
        for start in range(0,len(atlas),512):
            sl=slice(start,min(start+512,len(atlas)))
            response=np.asarray(x[np.ix_(rows,atlas[sl])],float);positive=(response>0).astype(float)
            den=np.maximum(positive.sum(0),1.)
            self.positive_coef[:,sl]=conditional_ridge(h,positive,response,ridge=1.)
            self.positive_mean[sl]=response.sum(0)/den
            anchor_response=self.donors[:,self.mapped[sl]].astype(float);anchor_positive=(anchor_response>0).astype(float)
            coef=conditional_ridge(ah,anchor_positive,anchor_response,ridge=1.)
            self.anchor_positive_delta[:,sl]=(coef-self.positive_coef[:,sl])*self.calibration_support[sl]
        self.audit.update(method='Conditional positive log1p response decoder',response_units='log1p normalized expression',
                          response_fit_rows=len(rows),response_fit_max_stage=float(stages[rows].max()),
                          limitations='Own scoring-scale adaptation; fitting scale alone does not guarantee DE improvement. Conditional positives omit zeros; detection model unchanged.')

    def positive_log_change(self,h0,h1,sl,target):
        delta=(h1-h0)@(self.positive_coef[:,sl]+self.positive_alpha*self.anchor_positive_delta[:,sl])
        observed=self.donors[:,self.mapped[sl]].astype(float)
        # Existing raw abundance factor cap applies downstream, as in the control.
        proposed=np.maximum(observed+delta,1e-8)
        raw=np.expm1(observed);future=np.expm1(np.minimum(proposed,np.log1p(10000.)))
        ratio=np.divide(future,raw,out=np.ones_like(future),where=raw>0)
        return np.log(np.maximum(ratio,1e-12))

    def positive_log_value(self,h1,sl,target):
        coef=self.positive_coef[:,sl];delta=self.anchor_positive_delta[:,sl]
        value=self.positive_mean[sl]+h1@coef-(self.positive_center[:,sl]*coef).sum(0)
        value+=self.positive_alpha*(h1@delta-(self.anchor_positive_center[:,sl]*delta).sum(0))
        return np.log(np.maximum(np.expm1(np.clip(value,1e-8,np.log1p(10000.))),1e-8))
