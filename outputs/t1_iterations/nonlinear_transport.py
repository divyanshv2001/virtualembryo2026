"""Past-only Sinkhorn pseudo-velocities and nonlinear autonomous latent flow.

A CPU adaptation of transport/dynamics ideas, not a TrajectoryNet reproduction.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from transfer_genes import apply_unique_factors
from robust_population import covariance_change


def coupling(a,b,regularization=.1,iterations=300):
    cost=cdist(a,b,'sqeuclidean')
    scale=max(float(np.median(cost)),1e-8)
    kernel=-cost/(regularization*scale)
    u=np.zeros(len(a)); v=np.zeros(len(b))
    for _ in range(iterations):
        u=-np.log(len(a))-logsumexp(kernel+v[None,:],axis=1)
        v=-np.log(len(b))-logsumexp(kernel+u[:,None],axis=0)
    plan=np.exp(kernel+u[:,None]+v[None,:])
    error=max(np.max(np.abs(plan.sum(1)-1/len(a))),np.max(np.abs(plan.sum(0)-1/len(b))))
    return plan,float(error)


class NonlinearTransport:
    def __init__(self,x,stages,cutoff,encoder,donors,base,base_indices,official_symbols,atlas_symbols,pair_cells=400,max_iter=250):
        if float(encoder['cutoff'])!=cutoff or str(encoder['alignment'])!='identity': raise ValueError('Require matching past encoder')
        self.encoder=encoder; self.cutoff=cutoff; self.donors=donors; self.base=base; self.base_indices=base_indices
        self.official_symbols=official_symbols; self.atlas_symbols=atlas_symbols; self.features=encoder['official_features']
        rows=np.flatnonzero(stages<=cutoff); times=np.unique(stages[rows])[-5:]
        if len(times)<3: raise ValueError('Three past stages required')
        latent=self.project(np.asarray(x[np.ix_(rows,encoder['features'])],dtype=float))
        self.center=latent.mean(0); self.scale=np.maximum(latent.std(0),.1)
        normalized=(latent-self.center)/self.scale
        inputs=[]; velocities=[]; errors=[]; rng=np.random.default_rng(2026092807)
        for a,b in zip(times[:-1],times[1:]):
            ia=np.flatnonzero(stages[rows]==a); ib=np.flatnonzero(stages[rows]==b)
            ia=np.sort(rng.choice(ia,min(pair_cells,len(ia)),replace=False)); ib=np.sort(rng.choice(ib,min(pair_cells,len(ib)),replace=False))
            za=normalized[ia]; zb=normalized[ib]; plan,error=coupling(za,zb)
            bary=(plan@zb)/np.maximum(plan.sum(1)[:,None],1e-12)
            inputs.append(za); velocities.append((bary-za)/(b-a)); errors.append(error)
        train=np.concatenate(inputs); truth=np.concatenate(velocities)
        self.velocity_cap=max(float(np.quantile(np.linalg.norm(truth,axis=1),.99)),1e-6)
        self.linear=Ridge(alpha=10.).fit(train,truth)
        self.nonlinear=MLPRegressor(hidden_layer_sizes=(64,64),activation='tanh',alpha=1.,max_iter=max_iter,
            random_state=20260928,early_stopping=True,validation_fraction=.15,n_iter_no_change=15).fit(train,truth)
        self.decoder=np.zeros((latent.shape[1],x.shape[1]))
        # Decode changes in log abundance; intercept cancels in a latent displacement.
        gram=normalized.T@normalized+10*np.eye(normalized.shape[1])
        for start in range(0,x.shape[1],256):
            values=np.asarray(x[np.ix_(rows,np.arange(start,min(start+256,x.shape[1])))],dtype=float)
            logabundance=np.log(np.expm1(values)+.1)
            self.decoder[:,start:start+values.shape[1]]=np.linalg.solve(gram,normalized.T@logabundance)
        self.audit={'past_stages':times.tolist(),'fit_max_stage':float(stages[rows].max()),'past_training_pairs':len(train),
            'coupling_marginal_errors':errors,'mlp_iterations':int(self.nonlinear.n_iter_),
            'mlp_training_loss':float(self.nonlinear.loss_),'iteration_limit':max_iter,
            'iteration_limit_reached':bool(self.nonlinear.n_iter_>=max_iter),'training_loss_is_forecast_validation':False,
            'velocity_norm_cap':self.velocity_cap,'method':'Autonomous MLP trained on barycentric Sinkhorn pseudo-velocities; not continuous normalizing flow or TrajectoryNet reproduction.'}
    def project(self,values):
        e=self.encoder
        return (np.clip((values-e['center'])/e['scale'],-10,10)-e['pca_mean'])@e['pca_components'].T
    def velocity(self,z,method):
        reg=self.nonlinear if method=='mlp' else self.linear
        v=reg.predict(z); norms=np.linalg.norm(v,axis=1)
        return v*np.minimum(1,self.velocity_cap/np.maximum(norms,1e-12))[:,None]
    def predict(self,target,strength=.5,method='mlp',factor_cap=1.25):
        if target<=self.cutoff or method not in ['mlp','ridge'] or not 0<=strength<=1: raise ValueError('Invalid forecast')
        z0=(self.project(self.donors[self.base_indices][:,self.features].astype(float))-self.center)/self.scale
        z=z0.copy(); step=(target-self.cutoff)/8
        for _ in range(8):
            k1=self.velocity(z,method); k2=self.velocity(z+step*k1/2,method)
            k3=self.velocity(z+step*k2/2,method); k4=self.velocity(z+step*k3,method)
            z += step*(k1+2*k2+2*k3+k4)/6
        delta=(z-z0)@self.decoder
        delta *= self.encoder['trusted'][self.base_indices,None]
        original=self.donors[:,self.features]
        for mix in [1.,.5,.25,.125,0.]:
            factors=np.exp(np.clip(strength*mix*delta,-np.log(factor_cap),np.log(factor_cap)))
            prediction,mapping=apply_unique_factors(self.base,self.official_symbols,self.atlas_symbols,factors)
            change=covariance_change(original,prediction[:,self.features])
            if change<=.4: break
        else: raise ValueError('Frozen base fails covariance guard')
        return prediction,self.base_indices.copy(),{**self.audit,'velocity_model':method,'requested_strength':strength,
            'used_strength':strength*mix,'factor_cap':factor_cap,'covariance_change_vs_reference':change,
            'mapping':mapping,'guard_applied':True,'future_representation_used':False}
    def save(self,path):
        payload={'center':self.center,'scale':self.scale,'decoder':self.decoder,'cutoff':np.array(self.cutoff),
            'ridge_coef':self.linear.coef_,'ridge_intercept':self.linear.intercept_,'velocity_cap':np.array(self.velocity_cap)}
        payload.update({f'mlp_weights_{i}':v for i,v in enumerate(self.nonlinear.coefs_)})
        payload.update({f'mlp_bias_{i}':v for i,v in enumerate(self.nonlinear.intercepts_)})
        np.savez_compressed(path,**payload)
