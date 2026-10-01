"""Past-fitted detection dependence; disclose margin changes from mass repair."""
import numpy as np
from scipy.special import ndtri
from sklearn.decomposition import PCA


class JointDetectionCoupling:
    def __init__(self, past, past_labels, features, mapped, seed=20261001):
        self.features=np.asarray(features); self.mapped=np.asarray(mapped); self.seed=seed
        detected=(past[:, self.features]>0).astype(np.float32)
        residual=detected.copy()
        for group in np.unique(past_labels):
            rows=past_labels==group
            residual[rows]-=detected[rows].mean(0)
        self.pca=PCA(n_components=min(8,len(residual)-1),random_state=seed).fit(residual)
        self.loading=self.pca.components_.T*np.sqrt(self.pca.explained_variance_)[None,:]
        self.noise=np.sqrt(np.maximum(residual.var(0)-np.square(self.loading).sum(1),1e-4))

    def apply(self, base, labels, strength):
        if not 0<=strength<=.5:raise ValueError('Frozen coupling range exceeded')
        if strength==0:return base.copy(),{'strength':0.,'backoff':1.}
        for backoff in (1.,.5,.25,0.):
            candidate=base.copy();rng=np.random.default_rng(self.seed)
            for group in np.unique(labels):
                rows=np.flatnonzero(labels==group)
                if len(rows)<20:continue
                latent=rng.standard_normal((len(rows),self.loading.shape[1]))
                proposals=latent@self.loading.T+rng.standard_normal((len(rows),len(self.features)))*self.noise
                for j,gene in enumerate(self.features):
                    old=base[rows,gene];n=int((old>0).sum())
                    if n==0 or n==len(rows):continue
                    # Stratified scores reproduce the current binary mask at zero coupling.
                    jitter=rng.random(len(rows))
                    ranks=np.argsort(np.argsort((old>0).astype(float)+jitter,kind='stable'),kind='stable')
                    existing=ndtri((ranks+.5)/len(rows))
                    proposed=proposals[:,j]/max(float(proposals[:,j].std()),1e-6)
                    s=strength*backoff
                    selected=np.argsort((1-s)*existing+s*proposed,kind='stable')[-n:]
                    values=np.sort(old[old>0])
                    # Fixed per-gene positives and counts BEFORE per-cell mass reconciliation.
                    candidate[rows,gene]=0
                    candidate[rows[selected],gene]=values
            before_projection_counts_exact=bool(np.array_equal((candidate>0).sum(0),(base>0).sum(0)))
            before_projection_margins_exact=bool(np.array_equal(np.sort(candidate[:,self.features],axis=0),np.sort(base[:,self.features],axis=0)))
            # Two-dimensional conservation constraints conflict: repair per-cell mapped mass,
            # explicitly measuring resulting changes to positive margins and gene means.
            oldmass=np.expm1(base[:,self.mapped].astype(float)).sum(1)
            values=np.expm1(candidate[:,self.mapped].astype(float));mass=values.sum(1)
            if np.any((mass<=0)&(oldmass>0)):continue
            ratio=np.divide(oldmass,mass,out=np.ones_like(oldmass),where=mass>0)
            candidate[:,self.mapped]=np.log1p(values*ratio[:,None]).astype(np.float32)
            mean_error=float(np.max(np.abs(candidate.mean(0)-base.mean(0))))
            counts_exact=bool(np.array_equal((candidate>0).sum(0),(base>0).sum(0)))
            repaired=np.expm1(candidate[:,self.mapped].astype(float)).sum(1)
            mass_error=float(np.max(np.abs(repaired-oldmass)/np.maximum(oldmass,1e-12)))
            if not counts_exact or mass_error>1e-5:continue
            # Frozen mean-perturbation bound avoids claiming a dependence-only improvement.
            if mean_error>.01:continue
            return candidate,{'strength':strength,'backoff':backoff,
                'detection_counts_exact':counts_exact,'preprojection_positive_margins_exact':before_projection_margins_exact,
                'preprojection_counts_exact':before_projection_counts_exact,
                'mapped_mass_relative_error':mass_error,'maximum_gene_mean_change':mean_error,
                'changed_detection_entries':int(((candidate>0)!=(base>0)).sum()),
                'positive_margins_after_mass_projection':'Not guaranteed; changes disclosed, all four metrics scored.',
                'scope':'Source annotations/dependence model are retrospective; no calibrated temporal dependence forecast.'}
        raise ValueError('No valid mass reconciliation, including identity backoff')


if __name__=='__main__':
    rng=np.random.default_rng(42)
    x=(rng.random((80,12))>.5)*rng.uniform(.2,.8,(80,12))
    x=x.astype(np.float32);labels=np.repeat(['a','b'],40)
    model=JointDetectionCoupling(x,labels,np.arange(8),np.arange(10))
    identity,_=model.apply(x,labels,0.)
    np.testing.assert_array_equal(identity,x)
    result,audit=model.apply(x,labels,.25)
    replay,_=model.apply(x,labels,.25)
    np.testing.assert_array_equal(result,replay)
    np.testing.assert_array_equal((result>0).sum(0),(x>0).sum(0))
    np.testing.assert_array_equal(result[:,10:],x[:,10:])
    assert audit['mapped_mass_relative_error']<1e-5
    print('Coupling checks passed: identity, seeded replay, detection counts, protected genes, mass')
