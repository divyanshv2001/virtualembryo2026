"""Past-only positive-marginal drift; rank coupling inspired, not ECC reproduction."""
from collections import Counter
import numpy as np
from scipy.stats import rankdata
from robust_population import covariance_change


class PositiveQuantileForecast:
    def __init__(self,x,stages,cutoff,donors,panel,symbols,features,min_positive=20):
        self.cutoff=cutoff;self.donors=donors;self.features=np.asarray(features)
        counts=Counter(symbols);lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
        self.mapped=np.array([i for i,s in enumerate(panel) if s in lookup])
        self.atlas=np.array([lookup[panel[i]] for i in self.mapped])
        past=np.unique(stages[stages<=cutoff])[-3:]
        if len(past)!=3:raise ValueError('Need three past snapshots')
        self.q=np.array([.05,.1,.25,.5,.75,.9,.95])
        self.slope=np.zeros((len(self.q),len(self.mapped)));self.support=np.zeros(len(self.mapped),int)
        for start in range(0,len(self.mapped),256):
            genes=self.atlas[start:start+256]
            blocks=[np.asarray(x[np.ix_(np.flatnonzero(stages==t),genes)],dtype=float) for t in past]
            support=np.stack([(b>0).sum(0) for b in blocks]).min(0)
            self.support[start:start+len(genes)]=support
            good=np.flatnonzero(support>=min_positive)
            if not len(good):continue
            quantiles=[]
            for block in blocks:
                a=np.expm1(block[:,good]);a[a==0]=np.nan
                quantiles.append(np.nanquantile(np.log(a),self.q,axis=0))
            quantiles=np.stack(quantiles)
            centered=past-past.mean()
            slope=(centered[:,None,None]*quantiles).sum(0)/(centered@centered)
            # A slope reversal at any quantile makes that quantile's extrapolation zero.
            consistent=np.diff(quantiles,axis=0).prod(axis=0)>=0
            slope*=consistent*support[good][None,:]/(support[good][None,:]+64)
            self.slope[:,start+good]=slope

    def predict(self,target,strength=1.,factor_cap=1.25,covariance_limit=.4):
        if target<=self.cutoff or not 0<=strength<=1 or not 1<=factor_cap<=2:
            raise ValueError('Invalid frozen forecast parameters')
        horizon=target-self.cutoff;candidate=self.donors.copy()
        for j,gene in enumerate(self.mapped):
            rows=np.flatnonzero(self.donors[:,gene]>0)
            if not len(rows) or not np.any(self.slope[:,j]):continue
            values=np.expm1(self.donors[rows,gene].astype(float))
            ranks=(rankdata(values,method='average')-.5)/len(values)
            change=np.clip(horizon*strength*self.slope[:,j],-np.log(factor_cap),np.log(factor_cap))
            proposed=values*np.exp(np.interp(ranks,self.q,change))
            order=np.argsort(values,kind='stable')
            proposed[order]=np.maximum.accumulate(proposed[order])
            candidate[rows,gene]=np.log1p(proposed).astype(np.float32)
        old=np.expm1(self.donors[:,self.mapped].astype(float)).sum(1)
        def normalize(values):
            abundance=np.expm1(values[:,self.mapped].astype(float));total=abundance.sum(1)
            ratio=np.divide(old,total,out=np.ones_like(old),where=total>0)
            result=values.copy();result[:,self.mapped]=np.log1p(abundance*ratio[:,None]).astype(np.float32)
            return result
        used=0.;pred=self.donors.copy()
        for mix in [1.,.5,.25,.125,0.]:
            proposal=normalize((self.donors+mix*(candidate-self.donors)).astype(np.float32))
            cov=covariance_change(self.donors[:,self.features],proposal[:,self.features])
            if cov<=covariance_limit:pred=proposal;used=mix;break
        protected=np.setdiff1d(np.arange(self.donors.shape[1]),self.mapped)
        if not np.array_equal(pred[:,protected],self.donors[:,protected]):raise ValueError('Protected genes changed')
        return pred,np.arange(len(pred)),{'method':'positive_quantile_drift','requested_strength':strength,
            'guard_mix':used,'factor_cap_before_mass_normalization':factor_cap,
            'covariance_change_vs_reference':float(cov),'eligible_genes':int((self.support>=20).sum()),
            'zero_mask_preserved':bool(np.array_equal(pred==0,self.donors==0)),
            'rank_scope':'Monotone positive-gene proposal; per-cell mass conservation can subsequently alter cross-cell ranks.',
            'assumption':'Historical positive-marginal quantile drift persists; no population or detection dynamics.',
            'literature_scope':'Adaptation inspired by marginal/rank separation in ECC; no calibrated weather ensemble or faithful ECC implementation.'}

    def save(self,path):
        np.savez_compressed(path,slope=self.slope,support=self.support,q=self.q,mapped=self.mapped,
            atlas=self.atlas,cutoff=self.cutoff,features=self.features)
