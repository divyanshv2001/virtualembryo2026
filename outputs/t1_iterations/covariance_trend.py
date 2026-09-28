"""Past-only latent covariance dynamics with OAS versus empirical covariance."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.covariance import OAS
from annotation_trend import AnnotationTrend
from robust_population import stable_slope,covariance_change


def matrix_power_psd(matrix,power):
    values,vectors=np.linalg.eigh((matrix+matrix.T)/2)
    return (vectors*np.maximum(values,1e-6)**power)@vectors.T


class CovarianceTrend(AnnotationTrend):
    def __init__(self,x,stages,source_labels,cutoff,donors,donor_labels,panel,symbols,features,estimator='oas'):
        super().__init__(x,stages,source_labels,cutoff,donors,donor_labels,panel,symbols,features)
        if estimator not in ['oas','empirical']:raise ValueError('Undeclared covariance estimator')
        self.estimator=estimator;past=np.flatnonzero(stages<=cutoff)
        fitrows=np.sort(np.random.default_rng(20260928).choice(past,min(3000,len(past)),replace=False))
        lookup=dict(zip(self.mapped,self.atlas));selected=np.array([lookup[i] for i in self.features])
        values=np.asarray(x[np.ix_(past,selected)],float);self.center=values.mean(0);self.scale=np.maximum(values.std(0),.1)
        self.pca=PCA(n_components=min(8,len(selected),len(fitrows)-1),random_state=20260928)
        self.pca.fit((np.asarray(x[np.ix_(fitrows,selected)],float)-self.center)/self.scale)
        z=self.pca.transform((values-self.center)/self.scale)
        labels=np.asarray(source_labels).astype(str)[past];past_stages=stages[past]
        residual=z.copy();groups=[]
        for time in np.unique(past_stages):
            for label in sorted(set(labels[past_stages==time])):
                rows=np.flatnonzero((past_stages==time)&(labels==label));groups.append(rows)
                residual[rows]-=z[rows].mean(0)
        recent=np.unique(past_stages)[-3:];covariances=[];errors=[];shrinkages=[]
        for time in recent:
            r=residual[past_stages==time]
            if estimator=='oas':
                fitted=OAS(assume_centered=True).fit(r);cov=fitted.covariance_;shrinkages.append(float(fitted.shrinkage_))
            else:cov=r.T@r/len(r);shrinkages.append(0.)
            covariances.append(cov)
            errors.append(np.sqrt(np.maximum(np.outer(np.diag(cov),np.diag(cov))+cov*cov,0)/len(r)))
        self.last_cov=covariances[-1]
        self.cov_slope=stable_slope(recent,np.stack(covariances).reshape(3,-1),np.stack(errors).reshape(3,-1)).reshape(self.last_cov.shape)
        self.cov_slope=(self.cov_slope+self.cov_slope.T)/2
        indices=np.searchsorted(past,fitrows);rfit=residual[indices]
        gram=rfit.T@rfit/len(rfit)+np.eye(rfit.shape[1]);self.decoder=np.zeros((rfit.shape[1],len(self.mapped)))
        for start in range(0,len(self.mapped),512):
            raw=np.asarray(x[np.ix_(past,self.atlas[start:start+512])],float)
            y=np.log(np.expm1(raw)+.1)
            for rows in groups:y[rows]-=y[rows].mean(0)
            self.decoder[:,start:start+raw.shape[1]]=np.linalg.solve(gram,rfit.T@y[indices]/len(rfit))
        donor_z=self.pca.transform((donors[:,self.features]-self.center)/self.scale)
        self.donor_residual=donor_z.copy()
        for label in sorted(set(self.donor_labels)):
            rows=np.flatnonzero(self.donor_labels==label);self.donor_residual[rows]-=donor_z[rows].mean(0)
        self.fit_audit={'estimator':estimator,'latent_dimensions':rfit.shape[1],'fit_cells':len(fitrows),
            'recent_stages':recent.tolist(),'covariance_shrinkages':shrinkages,
            'scope':'Pooled within-type residual covariance; OAS Gaussian iid assumptions do not establish independent embryos or temporal extrapolation. Ridge decoder and bounded coloring are adaptations; variogram is still evaluated directly.'}

    def predict_covariance(self,target,strength):
        if target<=self.cutoff or strength not in [.25,.5,1.]:raise ValueError('Undeclared covariance forecast')
        old_mass=np.expm1(self.donors[:,self.mapped].astype(float)).sum(1)
        half=matrix_power_psd(self.last_cov,.5);inverse=matrix_power_psd(self.last_cov,-.5)
        proposed=self.last_cov+(target-self.cutoff)*self.cov_slope
        relative=inverse@proposed@inverse
        values,vectors=np.linalg.eigh((relative+relative.T)/2)
        bounded=np.clip(values,.5,2.)
        coloring=(vectors*np.sqrt(bounded))@vectors.T
        transform=inverse@coloring@half
        delta=self.donor_residual@(transform-np.eye(len(transform)))@self.decoder
        for backoff in [1.,.5,.25,.125,0.]:
            factors=np.exp(np.clip(strength*backoff*delta,-np.log(1.25),np.log(1.25)))
            abundance=np.expm1(self.donors[:,self.mapped].astype(float))*factors
            abundance*=np.divide(old_mass,abundance.sum(1),out=np.ones_like(old_mass),where=abundance.sum(1)>0)[:,None]
            pred=self.donors.copy();pred[:,self.mapped]=np.log1p(abundance).astype(np.float32)
            cov=covariance_change(self.donors[:,self.features],pred[:,self.features])
            if cov<=.4:break
        return pred,np.arange(len(pred)),{'method':'latent_covariance_drift','requested_strength':strength,
            'backoff':backoff,'relative_eigenvalues_before_clip':values.tolist(),'relative_eigenvalues_after_clip':bounded.tolist(),
            'covariance_change_vs_reference':cov,'fit_audit':self.fit_audit,'fixed_cell_counts':True,
            'warning':'Zero mask and mapped mass preserved; full-gene means and covariance can change after projection.'}

    def save(self,path):
        np.savez_compressed(path,last_cov=self.last_cov,cov_slope=self.cov_slope,decoder=self.decoder,
            center=self.center,scale=self.scale,pca_components=self.pca.components_,pca_mean=self.pca.mean_,
            donor_residual=self.donor_residual,mapped=self.mapped,atlas=self.atlas,cutoff=self.cutoff)
