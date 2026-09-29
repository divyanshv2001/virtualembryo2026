"""CPU VAE/neural-ODE/Sinkhorn adaptation with past-only full-gene decoding."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint
from geomloss import SamplesLoss
from transfer_genes import apply_unique_factors
from robust_population import covariance_change


class LatentModel(nn.Module):
    def __init__(self,features,dimensions=16):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(features,128),nn.ReLU())
        self.mean=nn.Linear(128,dimensions);self.std=nn.Linear(128,dimensions)
        self.decoder=nn.Sequential(nn.Linear(dimensions,128),nn.ReLU(),nn.Linear(128,features))
        self.drift=nn.Sequential(nn.Linear(dimensions,64),nn.ReLU(),nn.Linear(64,dimensions))

    def encode(self,x):
        hidden=self.encoder(x)
        return self.mean(hidden),F.softplus(self.std(hidden))+1e-4

    def sample(self,x):
        mean,std=self.encode(x)
        return mean+std*torch.randn_like(mean)

    def velocity(self,time,z):
        velocity=self.drift(z)
        norm=torch.linalg.vector_norm(velocity,dim=1,keepdim=True)
        return velocity*torch.clamp(6/torch.clamp(norm,min=1e-6),max=1.)

    def trajectory(self,z,times):
        return odeint(self.velocity,z,times,method='euler',options={'step_size':.25})


class NeuralODEForecast:
    def __init__(self,x,stages,cutoff,donors,panel,symbols,official_features,beta=1.,
                 pretrain_steps=200,joint_steps=1000,batch=32,truth_batch=200,
                 checkpoint=None,event=None,resume=False):
        if beta not in [0.,1.] or pretrain_steps<1 or joint_steps<1:raise ValueError('Invalid frozen neural configuration')
        torch.set_num_threads(2);torch.manual_seed(20260928)
        self.cutoff=cutoff;self.donors=donors;self.panel=panel;self.symbols=symbols
        self.features=np.asarray(official_features,int)
        counts=Counter(symbols);lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
        self.mapped=np.array([i for i,s in enumerate(panel) if s in lookup]);self.atlas=np.array([lookup[panel[i]] for i in self.mapped])
        if any(panel[i] not in lookup for i in self.features):raise ValueError('Unmapped neural feature')
        atlas_features=np.array([lookup[panel[i]] for i in self.features])
        past=np.flatnonzero(stages<=cutoff);times=np.unique(stages[past])
        if len(times)<3 or not np.allclose(np.diff(times),.25):raise ValueError('Contiguous quarter-day past stages required')
        values=np.asarray(x[np.ix_(past,atlas_features)],np.float32)
        self.center=values.mean(0);self.scale=np.maximum(values.std(0),.1)
        normalized=torch.tensor((values-self.center)/self.scale)
        groups=[normalized[stages[past]==time] for time in times]
        self.net=LatentModel(len(self.features),min(16,len(self.features)))
        self.history=[];checkpoint=Path(checkpoint) if checkpoint else None
        loaded=None
        if resume and checkpoint and checkpoint.exists():
            loaded=torch.load(checkpoint,weights_only=False,map_location='cpu')
            if loaded['beta']!=beta or loaded['cutoff']!=cutoff:raise ValueError('Checkpoint scientific configuration mismatch')
            self.net.load_state_dict(loaded['net']);torch.set_rng_state(loaded['rng'])
            self.history=loaded['history']
        def emit(kind,**kwargs):
            if event:event(kind,cutoff=cutoff,beta=beta,**kwargs)
        def save(phase,step,optimizer):
            if checkpoint:
                torch.save({'net':self.net.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state(),
                    'phase':phase,'step':step,'history':self.history,'cutoff':cutoff,'beta':beta},checkpoint)
        optimizer=torch.optim.Adam(list(self.net.encoder.parameters())+list(self.net.mean.parameters())+
            list(self.net.std.parameters())+list(self.net.decoder.parameters()),lr=.001,betas=(.95,.99))
        start=0
        if loaded:
            start=loaded['step'] if loaded['phase']=='pretrain' else pretrain_steps
            if loaded['phase']=='pretrain':optimizer.load_state_dict(loaded['optimizer'])
        for step in range(start,pretrain_steps):
            source=normalized[torch.randint(len(normalized),(min(128,len(normalized)),))]
            recon=self.net.decoder(self.net.sample(source));loss=F.mse_loss(recon,source)
            if not torch.isfinite(loss):raise ValueError('Nonfinite pretrain loss')
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.net.parameters(),5.);optimizer.step()
            if (step+1)%100==0 or step+1==pretrain_steps:
                self.history.append({'phase':'pretrain','step':step+1,'loss':float(loss.detach())})
                save('pretrain',step+1,optimizer);emit('neural_training_checkpoint',phase='pretrain',step=step+1,loss=float(loss.detach()))
        optimizer=torch.optim.Adam(self.net.parameters(),lr=.001,betas=(.95,.99))
        start=0
        if loaded and loaded['phase']=='joint':start=loaded['step'];optimizer.load_state_dict(loaded['optimizer'])
        physical_times=torch.tensor(times-times[0],dtype=torch.float32)
        sinkhorn=SamplesLoss('sinkhorn',p=2,blur=.05,scaling=.5,debias=True,backend='tensorized')
        for step in range(start,joint_steps):
            source=groups[0][torch.randint(len(groups[0]),(batch,))]
            trajectory=self.net.trajectory(self.net.sample(source),physical_times)
            observed_loss=torch.zeros(());dynamic_loss=torch.zeros(())
            for k,group in enumerate(groups):
                truth=group[torch.randint(len(group),(truth_batch,))]
                observed_loss=observed_loss+sinkhorn(self.net.decoder(trajectory[k]),truth)/len(groups)
                if beta:
                    latent_truth=self.net.sample(group[torch.randint(len(group),(batch,))])
                    dynamic_loss=dynamic_loss+sinkhorn(trajectory[k],latent_truth)/len(groups)
            loss=observed_loss+beta*dynamic_loss
            if not torch.isfinite(loss):raise ValueError('Nonfinite joint loss')
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.net.parameters(),5.);optimizer.step()
            if (step+1)%100==0 or step+1==joint_steps:
                self.history.append({'phase':'joint','step':step+1,'loss':float(loss.detach()),
                    'observation_loss':float(observed_loss.detach()),'dynamic_loss':float(dynamic_loss.detach())})
                save('joint',step+1,optimizer);emit('neural_training_checkpoint',phase='joint',step=step+1,loss=float(loss.detach()))
        self.net.eval()
        with torch.no_grad():z=self.net.encode(normalized)[0].numpy().astype(float)
        self.latent_center=z.mean(0);centered=z-self.latent_center
        gram=centered.T@centered/len(centered)+np.eye(z.shape[1])
        self.full_decoder=np.zeros((z.shape[1],len(symbols)))
        for start in range(0,len(self.atlas),512):
            genes=self.atlas[start:start+512]
            raw=np.asarray(x[np.ix_(past,genes)],float);y=np.log(np.expm1(raw)+.1)
            coef=np.linalg.solve(gram,centered.T@y/len(centered))
            coef[:,(raw>0).sum(0)<20]=0
            self.full_decoder[:,genes]=coef
        self.audit={'method':'VAE neural ODE with Sinkhorn observation and latent dynamics losses',
            'beta':beta,'fit_cells':len(past),'feature_count':len(self.features),'fit_max_stage':float(stages[past].max()),
            'past_stages':times.tolist(),'pretrain_steps':pretrain_steps,'joint_steps':joint_steps,
            'batch':batch,'truth_batch':truth_batch,'training_history':self.history,
            'scope':'Author-architecture adaptation, not scNODE reproduction: normalized feature scaling, minibatch reconstruction, bounded velocity/gradient, fixed physical Euler step, ridge full-gene donor-relative decoding. No annotation labels or future expression are model inputs. All permitted prepared rows are available to training; prepared cohort is sampled, not the complete raw atlas.',
            'surrogate_is_final_score':False,'model_selection':'Predeclared final iteration; no target-based checkpoint selection.'}

    def predict(self,target,strength):
        if target<=self.cutoff or strength not in [.25,.5,1.]:raise ValueError('Invalid neural forecast')
        with torch.no_grad():
            inputs=torch.tensor((self.donors[:,self.features]-self.center)/self.scale)
            z0=self.net.encode(inputs)[0]
            # Conditional posterior mean removes forecast-seed selection; train is stochastic.
            trajectory=self.net.trajectory(z0,torch.tensor([0.,target-self.cutoff]))
            delta=(trajectory[-1]-z0).numpy().astype(float)@self.full_decoder
        if not np.isfinite(delta).all():raise ValueError('Nonfinite decoded drift')
        for backoff in [1.,.5,.25,.125,0.]:
            factors=np.exp(np.clip(strength*backoff*delta,-np.log(1.25),np.log(1.25)))
            pred,mapping=apply_unique_factors(self.donors,self.panel,self.symbols,factors)
            change=covariance_change(self.donors[:,self.features],pred[:,self.features])
            if change<=.4:break
        return pred,np.arange(len(pred)),{**self.audit,'requested_strength':strength,'backoff':backoff,
            'covariance_change_vs_reference':change,'mapping':mapping,'forecast_latent':'Conditional mean, no sampled seed selection.',
            'zero_mask_preserved':bool(np.array_equal(pred==0,self.donors==0))}

    def save(self,path):
        torch.save(self.net.state_dict(),Path(path).with_suffix('.pt'))
        np.savez_compressed(path,center=self.center,scale=self.scale,full_decoder=self.full_decoder,
            features=self.features,cutoff=self.cutoff)
