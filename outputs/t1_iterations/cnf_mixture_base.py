"""Fixed earliest-past Gaussian mixture base for likelihood flows."""
import math
import hashlib
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from cnf_density_flow import DensityFlowNet


def fit_earliest_prior(z,stages,cutoff,components):
    if components not in [1,8]:raise ValueError('Undeclared mixture components')
    allowed=np.flatnonzero(stages<=cutoff)
    if not len(allowed):raise ValueError('No permitted past rows')
    earliest=float(np.min(stages[allowed]));rows=np.flatnonzero(stages==earliest)
    values=np.asarray(z[rows],dtype=np.float64)
    if len(rows)<2*components or not np.isfinite(values).all():raise ValueError('Invalid earliest base rows')
    model=GaussianMixture(n_components=components,covariance_type='full',reg_covar=.01,n_init=1,random_state=20260928,max_iter=100).fit(values)
    if not model.converged_:raise ValueError('Mixture base did not converge in declared100steps')
    return {'weights':model.weights_,'means':model.means_,'precision_cholesky':model.precisions_cholesky_,
            'rows':rows,'earliest_stage':earliest,'iterations':model.n_iter_,'lower_bound':model.lower_bound_},model


class MixtureBaseDensityFlowNet(DensityFlowNet):
    def __init__(self,basis,pca_center,cutoff,origin,prior):
        super().__init__(basis,pca_center,cutoff,origin)
        weights=np.asarray(prior['weights'],np.float32);means=np.asarray(prior['means'],np.float32);precision=np.asarray(prior['precision_cholesky'],np.float32)
        if means.shape!=(len(weights),len(basis)) or precision.shape!=(len(weights),len(basis),len(basis)):
            raise ValueError('Prior dimension mismatch')
        if not all(np.isfinite(v).all() for v in [weights,means,precision]) or (weights<=0).any() or not np.isclose(weights.sum(),1.):
            raise ValueError('Invalid base parameters')
        if (np.diagonal(precision,axis1=1,axis2=2)<=0).any():raise ValueError('Invalid precision diagonal')
        self.base_signature=hashlib.sha256(weights.tobytes()+means.tobytes()+precision.tobytes()).hexdigest()
        self.register_buffer('base_means',torch.tensor(means))
        self.register_buffer('base_precision',torch.tensor(precision))
        constants=np.log(weights)+np.log(np.diagonal(precision,axis1=1,axis2=2)).sum(1)-.5*len(basis)*math.log(2*math.pi)
        self.register_buffer('base_constants',torch.tensor(constants,dtype=torch.float32))
    def log_base(self,z):
        delta=z[:,None,:]-self.base_means[None,:,:]
        whitened=torch.einsum('nkd,kde->nke',delta,self.base_precision)
        return torch.logsumexp(self.base_constants[None,:]-.5*whitened.square().sum(2),dim=1)
