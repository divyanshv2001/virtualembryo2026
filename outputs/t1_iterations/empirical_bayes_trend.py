"""Normal-mixture slope shrinkage adaptation; not ashr or embryo inference."""
import numpy as np
from scipy.special import logsumexp, ndtr
from annotation_trend import AnnotationTrend


def shrink_normal_effects(effect,se):
    effect=np.asarray(effect,float);se=np.asarray(se,float)
    if effect.shape!=se.shape or not np.isfinite(effect).all() or not np.isfinite(se).all() or (se<=0).any():
        raise ValueError('Finite effects and strictly positive standard errors required')
    scales=np.array([0.,.01,.03,.1,.3,1.,3.])
    variance=se[:,None]**2+scales[None,:]**2
    loglik=-.5*(np.log(2*np.pi*variance)+effect[:,None]**2/variance)
    pi=np.ones(len(scales))/len(scales);history=[];converged=False
    penalty=np.array([9.,0.,0.,0.,0.,0.,0.])
    for iteration in range(1000):
        lp=loglik+np.log(np.maximum(pi,1e-300))
        normalizer=logsumexp(lp,axis=1)
        weights=np.exp(lp-normalizer[:,None])
        objective=float(normalizer.sum()+9*np.log(max(pi[0],1e-300)))
        history.append(objective)
        updated=(weights.sum(0)+penalty)/(len(effect)+penalty.sum())
        if np.max(np.abs(updated-pi))<1e-7:
            pi=updated;converged=True;break
        pi=updated
    lp=loglik+np.log(np.maximum(pi,1e-300));weights=np.exp(lp-logsumexp(lp,axis=1)[:,None])
    shrink=scales[None,:]**2/variance
    means=effect[:,None]*shrink
    sd=np.sqrt(se[:,None]**2*shrink)
    positive=ndtr(np.divide(means[:,1:],sd[:,1:]))
    null=weights[:,0]
    p_le_zero=null+(weights[:,1:]*(1-positive)).sum(1)
    p_ge_zero=null+(weights[:,1:]*positive).sum(1)
    return (weights*means).sum(1),np.minimum(p_le_zero,p_ge_zero),{
        'mixture_scales':scales.tolist(),'mixture_weights':pi.tolist(),'iterations':iteration+1,
        'converged':converged,'penalized_objective_monotone':bool(np.all(np.diff(history)>=-1e-6)),
        'null_penalty':9,'scope':'Fixed normal-mixture grid and penalized EM; conditional cell-based uncertainty, not calibrated embryo false-sign probabilities.'}


class EmpiricalBayesTrend(AnnotationTrend):
    def __init__(self,x,stages,source_labels,cutoff,donors,donor_labels,panel,symbols,features,error_multiplier=1.):
        super().__init__(x,stages,source_labels,cutoff,donors,donor_labels,panel,symbols,features)
        if error_multiplier not in [1.,2.]:raise ValueError('Outside declared uncertainty ablation')
        recent=np.unique(stages[stages<=cutoff])[-3:];center=recent-recent.mean();denom=(center*center).sum()
        effects=np.zeros_like(self.positive_slope);errors=np.ones_like(effects);valid=np.zeros_like(effects,dtype=bool)
        labels=np.asarray(source_labels).astype(str)
        for k,label in enumerate(self.types):
            groups=[np.flatnonzero((stages==t)&(labels==label)) for t in recent]
            if min(map(len,groups))<20:continue
            for start in range(0,len(self.mapped),256):
                genes=self.atlas[start:start+256];means=[];ses=[];counts=[]
                for rows in groups:
                    a=np.expm1(np.asarray(x[np.ix_(rows,genes)],float));n=(a>0).sum(0);counts.append(n)
                    m=np.divide(a.sum(0),n,out=np.zeros(len(genes)),where=n>0)
                    second=np.divide((a*a).sum(0),n,out=np.zeros(len(genes)),where=n>0)
                    means.append(np.log(m+.1));ses.append(np.sqrt(np.maximum(second-m*m,0)/np.maximum(n,1))/(m+.1))
                means=np.stack(means);ses=np.stack(ses)
                beta=center@means/denom
                residual=means-means.mean(0)-center[:,None]*beta
                # Temporal lack of linear fit is additional uncertainty, not independent replication.
                se=np.sqrt(((center[:,None]*ses)**2).sum(0)/denom**2+(residual**2).sum(0)/denom)
                effects[k,start:start+len(genes)]=beta
                errors[k,start:start+len(genes)]=np.maximum(se,1e-3)*error_multiplier
                valid[k,start:start+len(genes)]=np.min(np.stack(counts),axis=0)>=10
        self.positive_slope[:]=0;self.sign_risk=np.ones_like(effects)
        if valid.any():
            shrunk,risk,audit=shrink_normal_effects(effects[valid],errors[valid])
            self.positive_slope[valid]=shrunk;self.sign_risk[valid]=risk
        else:audit={'converged':True,'no_supported_effects':True}
        self.fit_audit={**audit,'supported_effects':int(valid.sum()),'error_multiplier':error_multiplier,
            'assumption':'Latest three past positive-mean slopes predict a full day; curvature included as uncertainty; no adjacent-sign suppression beyond optional posterior sign gate.'}

    def predict_bayes(self,target,sign_threshold=None):
        if sign_threshold not in [None,.1,.25]:raise ValueError('Outside frozen sign thresholds')
        original=self.positive_slope
        if sign_threshold is not None:self.positive_slope=original*(self.sign_risk<=sign_threshold)
        try:
            pred,indices,audit=super().predict(target,1.,0.,.02)
            return pred,indices,{**audit,'method':'normal_mixture_slope_shrinkage','sign_threshold':sign_threshold,'fit_audit':self.fit_audit}
        finally:self.positive_slope=original

    def save(self,path):
        super().save(path)
        with np.load(path) as source:fields={k:source[k] for k in source.files}
        np.savez_compressed(path,**fields,sign_risk=self.sign_risk)
