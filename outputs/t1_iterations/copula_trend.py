"""Positive-margin rank coupling to past-fitted covariance forecast proposals."""
import numpy as np
from covariance_trend import CovarianceTrend
from robust_population import covariance_change


def couple_positive_margins(original,scores):
    if original.shape!=scores.shape:raise ValueError('Mismatched margins')
    positive=original>0
    ordered=np.sort(np.where(positive,original,np.inf),axis=0,kind='stable')
    ranking=np.argsort(np.where(positive,scores,np.inf),axis=0,kind='stable')
    result=np.zeros_like(original)
    np.put_along_axis(result,ranking,ordered,axis=0)
    result[~positive]=0
    return result


class CopulaTrend(CovarianceTrend):
    def predict_copula(self,target,strength):
        proposal,_,proposal_audit=self.predict_covariance(target,strength)
        original=self.donors[:,self.mapped].astype(float)
        reordered=original.copy();changed=0
        for label in sorted(set(self.donor_labels)):
            rows=np.flatnonzero(self.donor_labels==label)
            if len(rows)<20:continue
            for start in range(0,len(self.mapped),256):
                columns=np.arange(start,min(start+256,len(self.mapped)))
                before=original[np.ix_(rows,columns)]
                after=couple_positive_margins(before,proposal[np.ix_(rows,self.mapped[columns])])
                changed+=int((before!=after).sum());reordered[np.ix_(rows,columns)]=after
        ratio=np.divide(np.expm1(reordered),np.expm1(original),out=np.ones_like(original),where=original>0)
        factors=np.clip(ratio,1/1.25,1.25)
        old_mass=np.expm1(original).sum(1)
        for backoff in [1.,.5,.25,.125,0.]:
            abundance=np.expm1(original)*np.exp(backoff*np.log(factors))
            total=abundance.sum(1)
            abundance*=np.divide(old_mass,total,out=np.ones_like(old_mass),where=total>0)[:,None]
            pred=self.donors.copy();pred[:,self.mapped]=np.log1p(abundance).astype(np.float32)
            cov=covariance_change(self.donors[:,self.features],pred[:,self.features])
            if cov<=.4:break
        return pred,np.arange(len(pred)),{'method':'positive_margin_covariance_rank_coupling',
            'requested_strength':strength,'backoff':backoff,'reordered_positive_values_before_projection':changed,
            'covariance_change_vs_reference':cov,'covariance_proposal':proposal_audit,
            'scope':'Per-type positive margins exactly preserved before factor clipping and per-cell mass conservation. These projections change final margins and dependence. Stable ties, fixed zero mask; adaptation, not faithful ECC or biological copula recovery.'}
