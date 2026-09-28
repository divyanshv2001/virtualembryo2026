"""Past-only kernel-embedding drift and entropy-regularized cell resampling.

Adaptation of kernel mean matching using a forecast embedding, not future data.
The RFF training surrogate never replaces the full-panel evaluator.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.special import softmax
from constrained_forecast import slope
from robust_population import covariance_change


def weight_objective(theta,features,target,penalty):
    weights = softmax(theta)
    mean = weights@features; residual = mean-target
    log_ratio = np.log(weights*len(weights))
    divergence = float(weights@log_ratio)
    loss = float(residual@residual+penalty*divergence)
    gradient = 2*weights*((features-mean)@residual)+penalty*weights*(log_ratio-divergence)
    return loss,gradient


class KernelPopulation:
    def __init__(self,x,stages,cutoff,encoder,donors,base,base_indices,dimensions=500):
        if float(encoder['cutoff']) != cutoff or str(encoder['alignment']) != 'identity':
            raise ValueError('Require the matching past-only identity encoder')
        if donors.shape != base.shape or len(base_indices) != len(donors): raise ValueError('Base panel mismatch')
        self.encoder = encoder; self.cutoff = cutoff; self.donors = donors; self.base = base
        self.base_indices = base_indices
        self.features = encoder['official_features']
        self.trusted = encoder['trusted'][base_indices]
        rows = np.flatnonzero(stages <= cutoff); recent = np.unique(stages[rows])[-5:]
        if len(recent) < 5: raise ValueError('Five past stages required')
        source = np.asarray(x[np.ix_(rows,encoder['features'])],dtype=float)
        latent = self.project(source)
        last = latent[stages[rows] == cutoff]
        subset = np.random.default_rng(2026092804).choice(len(last),min(500,len(last)),replace=False)
        median = max(float(np.median(pdist(last[subset],metric='sqeuclidean'))),1e-6)
        rng = np.random.default_rng(2026092805)
        scales = np.resize(np.array([.25,.5,1.,2.,4.]),dimensions)
        gamma = 1/(median*scales**2)
        self.frequency = rng.normal(size=(latent.shape[1],dimensions))*np.sqrt(2*gamma)[None,:]
        self.phase = rng.uniform(0,2*np.pi,dimensions); self.median_distance = median
        embedding = self.embed_latent(latent)
        self.times = recent
        self.stage_means = np.stack([embedding[stages[rows] == t].mean(0) for t in recent])
        self.anchor_mean = self.embed_official(donors).mean(0)
        self.candidate_features = self.embed_official(base)

    def project(self,values):
        e = self.encoder
        scaled = np.clip((values-e['center'])/e['scale'],-10,10)
        return (scaled-e['pca_mean'])@e['pca_components'].T

    def embed_latent(self,latent):
        return np.sqrt(2/len(self.phase))*np.cos(latent@self.frequency+self.phase)

    def embed_official(self,values):
        return self.embed_latent(self.project(values[:,self.features].astype(float)))

    def predict(self,target,strength=.5,penalty=.01,window=5):
        if target <= self.cutoff or not 0 <= strength <= 1 or penalty <= 0 or window not in [3,5]:
            raise ValueError('Invalid frozen kernel configuration')
        drift = slope(self.times[-window:],self.stage_means[-window:])
        desired_mean = self.anchor_mean+(target-self.cutoff)*strength*drift
        f = self.candidate_features
        fit = minimize(weight_objective,np.zeros(len(f)),args=(f,desired_mean,penalty),jac=True,
            method='L-BFGS-B',bounds=[(-np.log(4),np.log(4))]*len(f),
            options={'maxiter':200,'ftol':1e-11,'gtol':1e-8})
        desired = softmax(fit.x)
        desired = np.where(self.trusted,desired,1/len(desired)); desired /= desired.sum()
        for mix in [1.,.5,.25,.125,0.]:
            weights = mix*desired+(1-mix)/len(desired)
            positions = np.minimum(np.searchsorted(np.cumsum(weights),(np.arange(len(f))+.5)/len(f)),len(f)-1)
            prediction = self.base[positions].copy()
            cov = covariance_change(self.donors[:,self.features],prediction[:,self.features])
            ess = float(1/np.sum(weights**2))
            if cov <= .4 and ess >= .6*len(f): break
        else: raise ValueError('Even unchanged incumbent failed the declared guard')
        uniform_loss = float(np.sum((f.mean(0)-desired_mean)**2))
        realized_loss = float(np.sum((f[positions].mean(0)-desired_mean)**2))
        return prediction,self.base_indices[positions],{
            'kernel_strength':strength,'kernel_penalty':penalty,'past_window':window,
            'optimizer_success':bool(fit.success),'optimizer_message':str(fit.message),'optimizer_iterations':int(fit.nit),
            'population_mix':mix,'effective_sample_size':ess,'covariance_change_vs_reference':cov,
            'surrogate_uniform_squared_error':uniform_loss,'surrogate_realized_squared_error':realized_loss,
            'surrogate_optimized_objective':float(fit.fun),'scoring_surrogate_used_as_final_score':False,
            'source_drift_norm':float(np.linalg.norm(drift)),'median_past_latent_squared_distance':self.median_distance,
            'guard_applied':True,'full_cells_preserved':True,'unique_base_cells':int(len(np.unique(positions)))}

    def save(self,path):
        np.savez_compressed(path,frequency=self.frequency,phase=self.phase,times=self.times,
            stage_means=self.stage_means,anchor_mean=self.anchor_mean,candidate_features=self.candidate_features,
            cutoff=np.array(self.cutoff),median_distance=np.array(self.median_distance))
