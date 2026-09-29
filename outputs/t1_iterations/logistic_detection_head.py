"""Regularized Bernoulli detection heads; no future expression fitting."""
from collections import Counter
import numpy as np
import torch
from scipy.special import expit,logit
from feature_panel_forecast import FeaturePanelForecast


def fit_logistic(h,y,ridge,steps=80):
    if ridge not in [.1,1.]:raise ValueError('Undeclared logistic ridge')
    design=np.column_stack([h,np.ones(len(h))]);n=len(h)
    coef=np.zeros((design.shape[1],y.shape[1]));coef[-1]=logit(np.clip(y.mean(0),1e-4,1-1e-4))
    penalty=np.full(design.shape[1],ridge);penalty[-1]=0.
    rate=1./(.25*np.linalg.eigvalsh(design.T@design/n).max()+ridge)
    initial=(np.logaddexp(0,design@coef)-y*(design@coef)).mean(0)
    for _ in range(steps):
        gradient=design.T@(expit(design@coef)-y)/n+penalty[:,None]*coef
        coef-=rate*gradient
    linear=design@coef;final=(np.logaddexp(0,linear)-y*linear).mean(0)+.5*(penalty[:,None]*coef*coef).sum(0)
    if not np.isfinite(coef).all() or np.any(final>initial+1e-10):raise ValueError('Invalid logistic optimization')
    gradient=design.T@(expit(linear)-y)/n+penalty[:,None]*coef
    return coef,{'initial_mean_objective':float(initial.mean()),'final_mean_objective':float(final.mean()),'max_abs_final_gradient':float(np.abs(gradient).max()),'steps':steps,'learning_rate':float(rate)}


class LogisticDetectionForecast(FeaturePanelForecast):
    def __init__(self,*args,detection_ridge=1.,emit=lambda **kw:None,**kwargs):
        super().__init__(*args,**kwargs)
        x,stages,cutoff=args[:3];panel=args[4];symbols=args[5]
        counts=Counter(symbols);lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
        rows=np.flatnonzero(stages<=cutoff);atlas=np.array([lookup[panel[i]] for i in self.mapped]);feat=np.array([lookup[panel[i]] for i in self.features])
        with torch.no_grad():h=self.net.encode(torch.tensor((np.asarray(x[np.ix_(rows,feat)])-self.center)/self.scale))[0].numpy().astype(float)-self.zcenter
        self.logistic_coef=np.zeros((h.shape[1]+1,len(atlas)));diagnostics=[]
        for start in range(0,len(atlas),512):
            sl=slice(start,min(start+512,len(atlas)));y=(np.asarray(x[np.ix_(rows,atlas[sl])])>0).astype(float)
            coef,audit=fit_logistic(h,y,detection_ridge);self.logistic_coef[:,sl]=coef;diagnostics.append(audit);emit(block_start=start,**audit)
        self.audit.update(method='Frozen CNF plus full conditional abundance and ridge Bernoulli-logistic detection',scope='MAST-inspired logistic detection adaptation, not MAST reproduction: latent covariates rather than experimental design/CDR, L2 rather than bayesglm prior, no DE inference or Gaussian positive likelihood.',detection_ridge=detection_ridge,logistic_diagnostics=diagnostics)

    def detection_probability(self,h,sl):
        return np.clip(expit(h@self.logistic_coef[:-1,sl]+self.logistic_coef[-1,sl]),1e-4,1-1e-4)
