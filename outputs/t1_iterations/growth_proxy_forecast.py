"""Bounded relative-growth composition sensitivity, not an OT dynamics model."""
import numpy as np
from robust_population import covariance_change


class GrowthProxyComposition:
    def __init__(self,donors,proliferation,p53_proxy,features,cutoff):
        self.donors=donors;self.features=features;self.cutoff=cutoff
        self.scores={'net':proliferation-p53_proxy,'proliferation':proliferation,'p53':-p53_proxy}
        if any(len(v)!=len(donors) or not np.isfinite(v).all() for v in self.scores.values()):raise ValueError('Invalid past growth proxies')

    def predict(self,target,kind,power,seed=20260928):
        if target<=self.cutoff or kind not in self.scores or power not in [0.,.5,1.]:raise ValueError('Invalid growth sensitivity')
        score=self.scores[kind];center=np.median(score);scale=max(float(score.std()),.1)
        standardized=(score-center)/scale
        offset=np.random.default_rng(seed).random();positions=(np.arange(len(score))+offset)/len(score)
        for backoff in [1.,.5,.25,.125,0.]:
            weights=np.exp(np.clip(power*backoff*(target-self.cutoff)*standardized,-np.log(2.),np.log(2.)))
            probability=weights/weights.sum();cdf=np.cumsum(probability);cdf[-1]=1.
            indices=np.searchsorted(cdf,positions,side='right')
            pred=self.donors[indices].copy()
            change=covariance_change(self.donors[:,self.features],pred[:,self.features])
            if change<=.4:break
        if not np.isfinite(pred).all():raise ValueError('Invalid composition prediction')
        return pred,indices,{'method':'Bounded growth-proxy composition resampling','kind':kind,'power':power,
            'backoff':backoff,'cutoff':self.cutoff,'target':target,'seed':seed,'covariance_change_vs_reference':change,
            'relative_weights_min':float(weights.min()),'relative_weights_max':float(weights.max()),
            'distinct_donors':len(np.unique(indices)),'whole_cells_preserved':True,
            'scope':'Expression-matched mouse proliferation/P53 proxies, not calibrated birth/death rates. Standardized bounded weights are a sensitivity test. Every predicted row is a complete observed donor row; fixed1500 output cells, per-selected-cell mass and protected genes preserved. No expression dynamics or unbalanced transport are fitted.'}
