"""Past-only conditional residual generator trained with exact latent RBF MMD.

CPU adaptation of moment matching, not a reproduction of unconditional GMMN.
"""
import numpy as np
from scipy.spatial.distance import cdist,pdist
from transfer_genes import apply_unique_factors
from robust_population import covariance_change,stable_slope
from collections import Counter


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


class ConditionedGenerator:
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
        self.parameters=[rng.normal(0,.1,(d+5,64)),np.zeros(64),rng.normal(0,.01,(64,d)),np.zeros(d)]
        averages=[np.zeros_like(v) for v in self.parameters];squares=[np.zeros_like(v) for v in self.parameters]
        validation=[]
        for index,(a,b) in enumerate(zip(groups[:-1],groups[1:])):
            validation.append((a[rng.choice(len(a),min(96,len(a)),replace=False)],b[rng.choice(len(b),min(96,len(b)),replace=False)],rng.normal(size=(min(96,len(a)),4)),times[index]))
        def validate():
            return float(np.mean([mmd_loss_gradient(a+.25*self.raw_velocity(a,noise,t)[0],b,self.gammas)[0] for a,b,noise,t in validation]))
        initial=validate();best=initial;best_parameters=[v.copy() for v in self.parameters];history=[{'step':0,'past_validation_mmd':initial}]
        for step in range(1,steps+1):
            pair=(step-1)%(len(groups)-1);a=groups[pair];b=groups[pair+1]
            source=a[rng.integers(len(a),size=batch)];truth=b[rng.integers(len(b),size=batch)];noise=rng.normal(size=(batch,4))
            velocity,hidden,inputs=self.raw_velocity(source,noise,times[pair]);generated=source+.25*velocity
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
        self.decoder_time_center=float(stages[rows].mean())
        design=np.column_stack([z,(stages[rows]-self.decoder_time_center)/.25])
        gram=design.T@design+10*np.eye(design.shape[1])
        self.probability_decoder=np.zeros((d+1,x.shape[1]));self.positive_decoder=np.zeros_like(self.probability_decoder)
        self.probability_intercept=np.zeros(x.shape[1]);self.positive_intercept=np.zeros(x.shape[1])
        self.effect_direction=np.zeros(x.shape[1]);self.detection_direction=np.zeros(x.shape[1])
        self.cutoff_positive_mean=np.zeros(x.shape[1]);self.cutoff_positive_support=np.zeros(x.shape[1],dtype=int)
        recent=times[-3:]
        for start in range(0,x.shape[1],256):
            end=min(start+256,x.shape[1]);values=np.asarray(x[np.ix_(rows,np.arange(start,end))],dtype=float)
            abundance=np.expm1(values);positive=abundance>0
            logs=np.zeros_like(abundance);np.log(abundance,out=logs,where=positive)
            self.probability_decoder[:,start:end]=np.linalg.solve(gram,design.T@positive)
            self.positive_decoder[:,start:end]=np.linalg.solve(gram,design.T@logs)
            self.probability_intercept[start:end]=positive.mean(0);self.positive_intercept[start:end]=logs.mean(0)
            means=[];errors=[];probabilities=[];prob_errors=[]
            for t in recent:
                block=abundance[stages[rows]==t];mean=block.mean(0)+.1;prob=(block>0).mean(0)
                means.append(np.log(mean));errors.append(block.std(0)/np.sqrt(len(block))/mean)
                probabilities.append(prob);prob_errors.append(np.sqrt(prob*(1-prob)/len(block)))
            self.effect_direction[start:end]=stable_slope(recent,np.array(means),np.array(errors))
            self.detection_direction[start:end]=stable_slope(recent,np.array(probabilities),np.array(prob_errors))
            block=abundance[stages[rows]==cutoff];support=(block>0).sum(0)
            self.cutoff_positive_support[start:end]=support
            self.cutoff_positive_mean[start:end]=np.divide(block.sum(0),support,out=np.zeros(end-start),where=support>0)
        counts=Counter(atlas_symbols);lookup={s:i for i,s in enumerate(atlas_symbols) if s and counts[s]==1}
        self.mapped=np.array([i for i,s in enumerate(official_symbols) if s in lookup],dtype=int)
        self.atlas_mapped=np.array([lookup[official_symbols[i]] for i in self.mapped],dtype=int)
        self.audit={'method':'Stage-conditioned noisy residual network trained on past adjacent-stage MMD; no faithful paper reproduction',
            'fit_max_stage':float(stages[rows].max()),'past_stages':times.tolist(),'training_steps':steps,'batch_size':batch,'training_objective':objective,
            'initial_past_validation_mmd':initial,'selected_past_validation_mmd':best,'training_history':history,
            'validation_scope':'Fixed cell groups from observed past stages, not an independent temporal holdout',
            'past_bandwidth':bandwidth,'future_representation_used':False,'surrogate_is_final_score':False}
    def project(self,values):
        e=self.encoder
        return (np.clip((values-e['center'])/e['scale'],-10,10)-e['pca_mean'])@e['pca_components'].T
    def raw_velocity(self,z,noise,time):
        inputs=np.column_stack([z,noise,np.full(len(z),(time-self.cutoff)/.25)]);w1,b1,w2,b2=self.parameters
        hidden=np.tanh(inputs@w1+b1)
        return hidden@w2+b2,hidden,inputs
    def predict(self,target,strength=.5,mode='freeze',direction_gate=False,decoder='legacy',detection_cap=.01):
        if target<=self.cutoff or mode not in ['freeze','extrapolate'] or decoder not in ['legacy','hurdle']:
            raise ValueError('Invalid forecast configuration')
        if not 0<=strength<=1 or not 0<=detection_cap<=.02:raise ValueError('Invalid strengths')
        z0=(self.project(self.donors[self.base_indices][:,self.features].astype(float))-self.center)/self.scale
        z=z0.copy();steps=max(1,int(np.ceil((target-self.cutoff)/.25)));dt=(target-self.cutoff)/steps
        rng=np.random.default_rng(2026092809)
        for step in range(steps):
            time=self.cutoff-.25 if mode=='freeze' else self.cutoff+step*dt
            velocity=self.raw_velocity(z,rng.normal(size=(len(z),4)),time)[0]
            norm=np.linalg.norm(velocity,axis=1);velocity*=np.minimum(1.,6/np.maximum(norm,1e-12))[:,None];z+=dt*velocity
        trust=self.encoder['trusted'][self.base_indices]
        if decoder=='legacy':
            delta=(z-z0)@self.decoder
            if direction_gate:delta=np.where(delta*self.effect_direction>0,delta,0.)
            delta*=trust[:,None]
            for mix in [1.,.5,.25,.125,0.]:
                prediction,mapping=apply_unique_factors(self.base,self.official_symbols,self.atlas_symbols,
                    np.exp(np.clip(strength*mix*delta,-np.log(1.25),np.log(1.25))))
                change=covariance_change(self.donors[:,self.features],prediction[:,self.features])
                if change<=.4:break
            activation=deletion=0
        else:
            start_features=np.column_stack([z0,np.full(len(z0),(self.cutoff-self.decoder_time_center)/.25)])
            end_features=np.column_stack([z,np.full(len(z),(target-self.decoder_time_center)/.25)])
            mass=np.expm1(self.base[:,self.mapped].astype(float)).sum(1)
            for mix in [1.,.5,.25,.125,0.]:
                abundance=np.expm1(self.base[:,self.mapped].astype(float));activation=deletion=0
                for start in range(0,len(self.mapped),512):
                    stop=min(start+512,len(self.mapped));genes=self.atlas_mapped[start:stop]
                    p0=np.clip(self.probability_intercept[genes]+start_features@self.probability_decoder[:,genes],.02,.98)
                    p1=np.clip(self.probability_intercept[genes]+end_features@self.probability_decoder[:,genes],.02,.98)
                    m0=np.clip((self.positive_intercept[genes]+start_features@self.positive_decoder[:,genes])/p0,-5,10)
                    m1=np.clip((self.positive_intercept[genes]+end_features@self.positive_decoder[:,genes])/p1,-5,10)
                    delta=m1-m0;probability=p1-p0
                    if direction_gate:
                        delta=np.where(delta*self.effect_direction[genes]>0,delta,0.)
                        probability=np.where(probability*self.detection_direction[genes]>0,probability,0.)
                    delta*=trust[:,None];probability*=trust[:,None]
                    delta*=strength*mix
                    probability=np.clip(strength*mix*probability,-detection_cap,detection_cap)
                    # Gene-indexed randomness stays identical across guard backoffs.
                    uniform=np.random.default_rng(2026092810+start).random(probability.shape)
                    block=abundance[:,start:stop];was_positive=block>0
                    add=(~was_positive)&(uniform<np.maximum(probability,0)/(1-p0))&(self.cutoff_positive_support[genes][None,:]>=10)
                    drop=was_positive&(uniform<np.maximum(-probability,0)/p0)
                    block*=np.exp(np.clip(delta,-np.log(1.25),np.log(1.25)))
                    proposed=np.broadcast_to(self.cutoff_positive_mean[genes],block.shape)*np.exp(np.clip(delta,-np.log(1.25),np.log(1.25)))
                    block[add]=proposed[add];block[drop]=0.;activation+=int(add.sum());deletion+=int(drop.sum())
                total=abundance.sum(1)
                # Empty proposals must fall back, not destroy measured library mass.
                empty=total<=0
                abundance[empty]=np.expm1(self.base[empty][:,self.mapped].astype(float));total=abundance.sum(1)
                ratio=np.divide(mass,total,out=np.ones_like(mass),where=total>0)
                prediction=self.base.copy();prediction[:,self.mapped]=np.log1p(abundance*ratio[:,None]).astype(np.float32)
                change=covariance_change(self.donors[:,self.features],prediction[:,self.features])
                if change<=.4:break
            mapping={'mapped_genes':len(self.mapped),'protected_genes':len(self.official_symbols)-len(self.mapped)}
        if change>.4:raise ValueError('Measured base fails covariance guard')
        return prediction,self.base_indices.copy(),{**self.audit,'time_mode':mode,'decoder':decoder,'direction_gate':direction_gate,
            'requested_strength':strength,'used_strength':strength*mix,'covariance_change_vs_reference':change,'mapping':mapping,
            'activated_values':activation,'deleted_values':deletion,'detection_cap':detection_cap,'guard_applied':True,
            'direction_gate_scope':'Proposal signs before mapped mass conservation; actual metric determines final direction.',
            'hurdle_scope':'Ridge detection and positive-log moment models on normalized abundance; not a raw-count likelihood or MAST reproduction.'}
    def save(self,path):
        payload={'center':self.center,'scale':self.scale,'decoder':self.decoder,'cutoff':np.array(self.cutoff),'gammas':self.gammas,'probability_decoder':self.probability_decoder,'positive_decoder':self.positive_decoder,
            'probability_intercept':self.probability_intercept,'positive_intercept':self.positive_intercept,
            'effect_direction':self.effect_direction,'detection_direction':self.detection_direction,
            'cutoff_positive_mean':self.cutoff_positive_mean,'cutoff_positive_support':self.cutoff_positive_support,
            'decoder_time_center':np.array(self.decoder_time_center)}
        payload.update({f'network_{i}':v for i,v in enumerate(self.parameters)})
        np.savez_compressed(path,**payload)
