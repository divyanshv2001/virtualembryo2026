"""Past-only detection and positive-abundance heads on frozen neural dynamics."""
from collections import Counter
import numpy as np
import torch
from neural_ode_forecast import LatentModel
from robust_population import covariance_change


class NeuralHurdleForecast:
    def __init__(self,x,stages,cutoff,donors,panel,symbols,net,center,scale,features):
        self.cutoff=cutoff;self.donors=donors;self.net=net;self.features=np.asarray(features,int)
        self.center=center;self.scale=scale
        counts=Counter(symbols);lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
        self.mapped=np.array([i for i,s in enumerate(panel) if s in lookup]);atlas=np.array([lookup[panel[i]] for i in self.mapped])
        rows=np.flatnonzero(stages<=cutoff)
        if not len(rows):raise ValueError('No permitted past rows')
        feat=np.array([lookup[panel[i]] for i in self.features])
        with torch.no_grad():z=net.encode(torch.tensor((np.asarray(x[np.ix_(rows,feat)])-center)/scale))[0].numpy().astype(float)
        self.zcenter=z.mean(0);h=z-self.zcenter;n=len(h)
        gram=h.T@h/n+np.eye(h.shape[1])
        g=len(atlas);d=h.shape[1]
        self.pmean=np.zeros(g);self.detection=np.zeros((d,g));self.positive_mean=np.zeros(g)
        self.positive_center=np.zeros((d,g));self.positive_coef=np.zeros((d,g));self.support=np.zeros(g,bool)
        for start in range(0,g,512):
            sl=slice(start,min(start+512,g));raw=np.asarray(x[np.ix_(rows,atlas[sl])],float)
            detected=(raw>0).astype(float);count=detected.sum(0);den=np.maximum(count,1)
            self.support[sl]=count>=20;self.pmean[sl]=count/n
            self.detection[:,sl]=np.linalg.solve(gram,h.T@detected/n)
            y=np.log(np.maximum(np.expm1(raw),1e-8))*detected
            mean=y.sum(0)/den;zc=h.T@detected/den
            variance=np.maximum((h*h).T@detected/den-zc*zc,0)
            covariance=h.T@y/den-zc*mean
            self.positive_mean[sl]=mean;self.positive_center[:,sl]=zc
            self.positive_coef[:,sl]=covariance/(variance+1.)
        self.detection[:,~self.support]=0;self.positive_coef[:,~self.support]=0
        self.net.eval()
        self.audit={'fit_cells':int(n),'fit_max_stage':float(stages[rows].max()),'supported_genes':int(self.support.sum()),
            'method':'Frozen VAE/neural ODE plus ridge1 detection probability and diagonal-shrunken positive-log abundance heads',
            'scope':'Adaptation and mechanistic ablation, not scNODE reproduction or calibrated biological detection probabilities. All heads fit permitted past cells only. Positive abundance uses diagonal conditional regression with ridge1, not a full covariance solve.'}

    def predict(self,target,mode,strength=1.,seed=20260928):
        if target<=self.cutoff or mode not in ['abundance','detection','joint'] or strength not in [.5,1.]:raise ValueError('Invalid hurdle forecast')
        with torch.no_grad():
            z0=self.net.encode(torch.tensor((self.donors[:,self.features]-self.center)/self.scale))[0]
            z1=self.net.trajectory(z0,torch.tensor([0.,target-self.cutoff]))[-1]
        h0=z0.numpy().astype(float)-self.zcenter;h1=z1.numpy().astype(float)-self.zcenter
        donor_mass=np.expm1(self.donors[:,self.mapped].astype(float)).sum(1)
        empty=donor_mass==0
        for backoff in [1.,.5,.25,.125,0.]:
            result=self.donors.copy();rng=np.random.default_rng(seed)
            for start in range(0,len(self.mapped),512):
                sl=slice(start,min(start+512,len(self.mapped)));genes=self.mapped[sl]
                original=np.expm1(self.donors[:,genes].astype(float));positive=original>0
                updated=original.copy();active=self.support[sl]
                if mode in ['abundance','joint']:
                    change=(h1-h0)@self.positive_coef[:,sl]
                    updated*=np.exp(np.clip(strength*backoff*change,-np.log(2.),np.log(2.)))
                if mode in ['detection','joint']:
                    p0=np.clip(self.pmean[sl]+h0@self.detection[:,sl],1e-4,1-1e-4)
                    p1=np.clip(self.pmean[sl]+h1@self.detection[:,sl],1e-4,1-1e-4)
                    change=np.clip(strength*backoff*(p1-p0),-.25,.25)*active
                    uniforms=rng.random(original.shape,dtype=np.float32)
                    added=(~positive)&(change>0)&(uniforms<change/(1-p0))
                    removed=positive&(change<0)&(uniforms<(-change)/p0)
                    # Subtract each gene's positive-cell latent centroid for its conditional intercept.
                    log_positive=self.positive_mean[sl]+h1@self.positive_coef[:,sl]-(self.positive_center[:,sl]*self.positive_coef[:,sl]).sum(0)
                    imputed=np.exp(np.clip(log_positive,-8.,np.log(10000.)))
                    updated[added]=imputed[added];updated[removed]=0
                updated[empty]=0
                result[:,genes]=np.log1p(updated).astype(np.float32)
            proposed=np.expm1(result[:,self.mapped].astype(float));new_mass=proposed.sum(1)
            # Restore a donor if stochastic removal erased its entire mapped library.
            erased=(new_mass==0)&(donor_mass>0)
            if erased.any():result[np.ix_(erased,self.mapped)]=self.donors[np.ix_(erased,self.mapped)]
            proposed=np.expm1(result[:,self.mapped].astype(float));new_mass=proposed.sum(1)
            ratio=np.divide(donor_mass,new_mass,out=np.ones_like(new_mass),where=new_mass>0)
            result[:,self.mapped]=np.log1p(proposed*ratio[:,None]).astype(np.float32)
            change=covariance_change(self.donors[:,self.features],result[:,self.features])
            if np.isfinite(result).all() and change<=.4:break
        if not np.isfinite(result).all() or (result<0).any():raise ValueError('Invalid hurdle output')
        return result,np.arange(len(result)),{**self.audit,'mode':mode,'strength':strength,'backoff':backoff,
            'seed':seed,'covariance_change_vs_reference':change,'newly_detected_entries':int(((self.donors==0)&(result>0)).sum()),
            'removed_detection_entries':int(((self.donors>0)&(result==0)).sum()),
            'abundance_factor_cap':2.,'detection_probability_delta_cap':.25,
            'common_uniform_policy':'Fixed seed reset across modes/strengths/backoffs; no seed selection.',
            'mapped_mass_conserved':True,'protected_genes_unchanged':True}
