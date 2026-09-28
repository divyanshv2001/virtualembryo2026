"""Past-only conditional residual generator trained with exact latent RBF MMD.

CPU adaptation of moment matching, not a reproduction of unconditional GMMN.
"""
import numpy as np
from scipy.spatial.distance import cdist,pdist
from transfer_genes import apply_unique_factors
from robust_population import covariance_change


def mmd_loss_gradient(generated,truth,gammas):
    n,m=len(generated),len(truth)
    xx=cdist(generated,generated,'sqeuclidean');xy=cdist(generated,truth,'sqeuclidean');tt=cdist(truth,truth,'sqeuclidean')
    loss=0.;gradient=np.zeros_like(generated)
    for gamma in gammas:
        a=np.exp(-gamma*xx);b=np.exp(-gamma*xy);c=np.exp(-gamma*tt)
        loss += a.mean()+c.mean()-2*b.mean()
        gradient += -4*gamma/n**2*(generated*a.sum(1)[:,None]-a@generated)+4*gamma/(n*m)*(generated*b.sum(1)[:,None]-b@truth)
    return float(loss/len(gammas)),gradient/len(gammas)


def training_loss_gradient(generated,truth,gammas,objective):
    loss,gradient=mmd_loss_gradient(generated,truth,gammas)
    if objective=='squared':return loss,gradient
    if objective!='sqrt':raise ValueError('Unknown objective')
    if loss<=1e-10:return 1e-5,np.zeros_like(gradient)
    root=np.sqrt(loss)
    return float(root),gradient/(2*root)


class DistributionGenerator:
    def __init__(self,x,stages,cutoff,encoder,donors,base,base_indices,official_symbols,atlas_symbols,steps=800,batch=64,objective='squared'):
        if objective not in ['squared','sqrt']:raise ValueError('Unknown objective')
        if float(encoder['cutoff'])!=cutoff or str(encoder['alignment'])!='identity': raise ValueError('Require matching past encoder')
        self.encoder=encoder;self.cutoff=cutoff;self.donors=donors;self.base=base;self.base_indices=base_indices
        self.official_symbols=official_symbols;self.atlas_symbols=atlas_symbols;self.features=encoder['official_features']
        rows=np.flatnonzero(stages<=cutoff);times=np.unique(stages[rows])[-5:]
        if len(times)<3 or not np.allclose(np.diff(times),.25): raise ValueError('Three contiguous quarter-day past stages required')
        latent=self.project(np.asarray(x[np.ix_(rows,encoder['features'])],dtype=float))
        self.center=latent.mean(0);self.scale=np.maximum(latent.std(0),.1);z=(latent-self.center)/self.scale
        groups=[z[stages[rows]==t] for t in times];d=z.shape[1];rng=np.random.default_rng(2026092808)
        bandwidth=max(float(np.median(pdist(z[rng.choice(len(z),min(200,len(z)),replace=False)],'sqeuclidean'))),1e-6)
        self.gammas=1/(bandwidth*np.array([.25,.5,1,2,4.])**2)
        self.parameters=[rng.normal(0,.1,(d+4,64)),np.zeros(64),rng.normal(0,.01,(64,d)),np.zeros(d)]
        averages=[np.zeros_like(v) for v in self.parameters];squares=[np.zeros_like(v) for v in self.parameters]
        validation=[]
        for a,b in zip(groups[:-1],groups[1:]):
            validation.append((a[rng.choice(len(a),min(96,len(a)),replace=False)],b[rng.choice(len(b),min(96,len(b)),replace=False)],rng.normal(size=(min(96,len(a)),4))))
        def validate():
            return float(np.mean([mmd_loss_gradient(a+.25*self.raw_velocity(a,noise)[0],b,self.gammas)[0] for a,b,noise in validation]))
        initial=validate();best=initial;best_parameters=[v.copy() for v in self.parameters];history=[{'step':0,'past_validation_mmd':initial}]
        for step in range(1,steps+1):
            pair=(step-1)%(len(groups)-1);a=groups[pair];b=groups[pair+1]
            source=a[rng.integers(len(a),size=batch)];truth=b[rng.integers(len(b),size=batch)];noise=rng.normal(size=(batch,4))
            velocity,hidden,inputs=self.raw_velocity(source,noise);generated=source+.25*velocity
            loss,gradient=training_loss_gradient(generated,truth,self.gammas,objective)
            output_gradient=.25*gradient+.002*velocity/(batch*d)
            w1,b1,w2,b2=self.parameters
            hidden_gradient=(output_gradient@w2.T)*(1-hidden**2)
            gradients=[inputs.T@hidden_gradient,hidden_gradient.sum(0),hidden.T@output_gradient,output_gradient.sum(0)]
            norm=np.sqrt(sum(np.sum(v*v) for v in gradients));clip=min(1.,5/max(norm,1e-12))
            for i,g in enumerate(gradients):
                g=g*clip;averages[i]=.9*averages[i]+.1*g;squares[i]=.999*squares[i]+.001*g*g
                self.parameters[i] -= .002*(averages[i]/(1-.9**step))/(np.sqrt(squares[i]/(1-.999**step))+1e-8)
            if step%100==0 or step==steps:
                measured=validate();history.append({'step':step,'past_validation_mmd':measured})
                if measured<best:best=measured;best_parameters=[v.copy() for v in self.parameters]
        self.parameters=best_parameters
        self.decoder=np.zeros((d,x.shape[1]));gram=z.T@z+10*np.eye(d)
        for start in range(0,x.shape[1],256):
            values=np.asarray(x[np.ix_(rows,np.arange(start,min(start+256,x.shape[1])))],dtype=float)
            self.decoder[:,start:start+values.shape[1]]=np.linalg.solve(gram,z.T@np.log(np.expm1(values)+.1))
        self.audit={'method':'Conditional noisy residual network trained on past adjacent-stage MMD; not unconditional GMMN reproduction',
            'fit_max_stage':float(stages[rows].max()),'past_stages':times.tolist(),'training_steps':steps,'batch_size':batch,'training_objective':objective,
            'initial_past_validation_mmd':initial,'selected_past_validation_mmd':best,'training_history':history,
            'validation_scope':'Fixed cell groups from observed past stages, not an independent temporal holdout',
            'past_bandwidth':bandwidth,'future_representation_used':False,'surrogate_is_final_score':False}
    def project(self,values):
        e=self.encoder
        return (np.clip((values-e['center'])/e['scale'],-10,10)-e['pca_mean'])@e['pca_components'].T
    def raw_velocity(self,z,noise):
        inputs=np.column_stack([z,noise]);w1,b1,w2,b2=self.parameters
        hidden=np.tanh(inputs@w1+b1)
        return hidden@w2+b2,hidden,inputs
    def predict(self,target,strength=.5,method='gmmn',factor_cap=1.25):
        if target<=self.cutoff or method not in ['gmmn','meanflow'] or not 0<=strength<=1: raise ValueError('Invalid forecast')
        z0=(self.project(self.donors[self.base_indices][:,self.features].astype(float))-self.center)/self.scale
        z=z0.copy();steps=max(1,int(np.ceil((target-self.cutoff)/.25)));dt=(target-self.cutoff)/steps
        rng=np.random.default_rng(2026092809)
        for _ in range(steps):
            noise=rng.normal(size=(len(z),4)) if method=='gmmn' else np.zeros((len(z),4))
            velocity=self.raw_velocity(z,noise)[0];norm=np.linalg.norm(velocity,axis=1)
            velocity*=np.minimum(1.,6/np.maximum(norm,1e-12))[:,None];z+=dt*velocity
        delta=(z-z0)@self.decoder;delta*=self.encoder['trusted'][self.base_indices,None]
        for mix in [1.,.5,.25,.125,0.]:
            factors=np.exp(np.clip(strength*mix*delta,-np.log(factor_cap),np.log(factor_cap)))
            prediction,mapping=apply_unique_factors(self.base,self.official_symbols,self.atlas_symbols,factors)
            change=covariance_change(self.donors[:,self.features],prediction[:,self.features])
            if change<=.4:break
        else:raise ValueError('Base fails covariance guard')
        return prediction,self.base_indices.copy(),{**self.audit,'forecast_noise':method=='gmmn','requested_strength':strength,
            'used_strength':strength*mix,'factor_cap':factor_cap,'covariance_change_vs_reference':change,'mapping':mapping,'guard_applied':True}
    def save(self,path):
        payload={'center':self.center,'scale':self.scale,'decoder':self.decoder,'cutoff':np.array(self.cutoff),'gammas':self.gammas}
        payload.update({f'network_{i}':v for i,v in enumerate(self.parameters)})
        np.savez_compressed(path,**payload)
