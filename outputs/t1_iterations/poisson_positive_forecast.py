"""Quasi-Poisson positive conditional means on normalized expression, not raw UMI likelihood."""
from collections import Counter
import numpy as np
import torch
from feature_panel_forecast import FeaturePanelForecast


def fit_positive_poisson(h,y,ridge=1.,steps=12,family='poisson'):
    if ridge not in [.1,1.]:raise ValueError('Undeclared positive ridge')
    if family not in ['poisson','gamma']:raise ValueError('Undeclared positive family')
    if not np.isfinite(y).all() or (y<0).any():raise ValueError('Invalid normalized abundance')
    design=np.column_stack([h,np.ones(len(h))]);d=design.shape[1]
    mask=(y>0).astype(float);counts=mask.sum(0);den=np.maximum(counts,1.)
    supported=counts>=20;penalty=np.array([ridge]*(d-1)+[0.])
    coef=np.zeros((d,y.shape[1]));coef[-1]=np.log(np.maximum(y.sum(0)/den,1e-8))
    pairs=(design[:,:,None]*design[:,None,:]).reshape(len(h),d*d)
    def objective(c):
        eta=design@c;mu=np.exp(np.clip(eta,-30,30))
        value=mu-y*eta if family=='poisson' else y/mu+eta
        return (value*mask).sum(0)/den+.5*(penalty[:,None]*c*c).sum(0)
    initial=objective(coef);current=initial.copy();iterations=0
    for iteration in range(steps):
        eta=design@coef;mu=np.exp(np.clip(eta,-30,30))
        residual=mu-y if family=='poisson' else 1-y/mu
        curvature=mu if family=='poisson' else y/mu
        gradient=design.T@(residual*mask)/den+penalty[:,None]*coef
        gradient[:,~supported]=0
        hessian=(pairs.T@(curvature*mask)/den).T.reshape(len(counts),d,d)
        hessian+=np.diag(penalty)[None,:,:]+1e-8*np.eye(d)[None,:,:]
        delta=np.linalg.solve(hessian,gradient.T[:,:,None])[:,:,0].T
        scale=np.ones(len(counts));accepted=coef.copy();next_objective=current.copy()
        pending=supported.copy()
        for _ in range(12):
            proposed=coef-delta*scale
            value=objective(proposed);good=pending&(value<=current+1e-10)&np.isfinite(value)
            accepted[:,good]=proposed[:,good];next_objective[good]=value[good]
            pending[good]=False
            if not pending.any():break
            scale[pending]*=.5
        coef=accepted;current=next_objective;iterations=iteration+1
        if np.max(np.abs(gradient))<1e-6:break
    if not np.isfinite(coef).all() or np.any(current>initial+1e-8):raise ValueError('Invalid Poisson mean optimization')
    final_mu=np.exp(np.clip(design@coef,-30,30))
    residual=final_mu-y if family=='poisson' else 1-y/final_mu
    gradient=design.T@(residual*mask)/den+penalty[:,None]*coef
    gradient[:,~supported]=0
    return coef,{'initial_mean_objective':float(initial.mean()),'final_mean_objective':float(current.mean()),
                 'max_supported_gradient':float(np.abs(gradient).max()),'iterations':iterations,
                 'supported_genes':int(supported.sum()),'family':family,
                 'loss_scope':'Positive-only quasi-Poisson mean objective on fractional expression; not raw-count likelihood.' if family=='poisson' else 'Positive continuous Gamma log-link mean objective, fixed unit shape; no raw-count likelihood or dispersion inference.'}


class PoissonPositiveForecast(FeaturePanelForecast):
    def __init__(self,*args,positive_ridge=1.,emit=lambda **kw:None,positive_family='poisson',**kwargs):
        super().__init__(*args,**kwargs)
        x,stages,cutoff=args[:3];panel,symbols=args[4:6]
        counts=Counter(symbols);lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
        rows=np.flatnonzero(stages<=cutoff);atlas=np.array([lookup[panel[i]] for i in self.mapped]);feat=[lookup[panel[i]] for i in self.features]
        with torch.no_grad():
            h=self.net.encode(torch.tensor((np.asarray(x[np.ix_(rows,feat)])-self.center)/self.scale))[0].numpy().astype(float)-self.zcenter
        self.poisson_coef=np.zeros((h.shape[1]+1,len(atlas)));diagnostics=[]
        for start in range(0,len(atlas),256):
            sl=slice(start,min(start+256,len(atlas)))
            y=np.expm1(np.asarray(x[np.ix_(rows,atlas[sl])],float))
            coef,audit=fit_positive_poisson(h,y,positive_ridge,family=positive_family);self.poisson_coef[:,sl]=coef
            diagnostics.append(audit);emit(block_start=start,**audit)
        self.audit.update(positive_loss='quasi-Poisson conditional mean' if positive_family=='poisson' else 'Gamma log-link conditional positive mean',positive_ridge=positive_ridge,poisson_diagnostics=diagnostics,
                          scope='Own conditional mean-loss ablation with fractional normalized expression. Not scVI, NB, raw-UMI likelihood or zero-truncated count model. Global detection/flow and prediction constraints fixed; past-only fit.')

    def positive_log_change(self,h0,h1,sl,target):
        return (h1-h0)@self.poisson_coef[:-1,sl]

    def positive_log_value(self,h1,sl,target):
        return h1@self.poisson_coef[:-1,sl]+self.poisson_coef[-1,sl]
