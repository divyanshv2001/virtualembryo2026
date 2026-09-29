"""Predeclared ridge sensitivity for conditional positive abundance heads."""
from collections import Counter
import numpy as np
import torch
from neural_hurdle_forecast import NeuralHurdleForecast


def conditional_ridge(h, positive, log_values, ridge=1.):
    """Gene-specific centered conditional regression, population scaling."""
    count=positive.sum(0);den=np.maximum(count,1.)
    zcenter=h.T@positive/den;mean=log_values.sum(0)/den
    pairs=(h[:,:,None]*h[:,None,:]).reshape(len(h),-1)
    second=(pairs.T@positive/den).T.reshape(len(count),h.shape[1],h.shape[1])
    covariance=second-np.einsum('ig,jg->gij',zcenter,zcenter)
    rhs=(h.T@log_values/den-zcenter*mean).T
    gram=covariance+ridge*np.eye(h.shape[1])[None,:,:]
    coefficient=np.linalg.solve(gram,rhs[:,:,None])[:,:,0].T
    coefficient[:,count<20]=0
    if not np.isfinite(coefficient).all():raise ValueError('Nonfinite conditional coefficients')
    return coefficient


class RidgeConditionalPositiveForecast(NeuralHurdleForecast):
    def __init__(self,x,stages,cutoff,donors,panel,symbols,net,center,scale,features,ridge=1.):
        if ridge not in [.1,.3,1.,3.]:raise ValueError('Unplanned ridge strength')
        self.conditional_ridge_strength=ridge
        super().__init__(x,stages,cutoff,donors,panel,symbols,net,center,scale,features)
        self.diagonal_positive_coef=self.positive_coef.copy()
        counts=Counter(symbols);lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
        rows=np.flatnonzero(stages<=cutoff);feat=np.array([lookup[panel[i]] for i in features])
        atlas=np.array([lookup[panel[i]] for i in self.mapped])
        with torch.no_grad():
            z=net.encode(torch.tensor((np.asarray(x[np.ix_(rows,feat)])-center)/scale))[0].numpy().astype(float)
        h=z-self.zcenter
        for start in range(0,len(atlas),512):
            sl=slice(start,min(start+512,len(atlas)))
            raw=np.asarray(x[np.ix_(rows,atlas[sl])],float);detected=(raw>0).astype(float)
            y=np.log(np.maximum(np.expm1(raw),1e-8))*detected
            self.positive_coef[:,sl]=conditional_ridge(h,detected,y,ridge=ridge)
        self.audit.update(conditional_ridge_strength=ridge,positive_regression='Full conditional covariance with predeclared ridge; support20. Own shrinkage sensitivity, not a published biological model reproduction.')
