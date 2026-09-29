"""Expanded encoder features with unchanged original covariance guard panel."""
import numpy as np
import torch
from neural_hurdle_forecast import systematic_bernoulli
from robust_population import covariance_change
from ridge_conditional_head import RidgeConditionalPositiveForecast


class FeaturePanelForecast(RidgeConditionalPositiveForecast):
    def __init__(self,x,stages,cutoff,donors,panel,symbols,net,center,scale,features,guard_features):
        super().__init__(x,stages,cutoff,donors,panel,symbols,net,center,scale,features,ridge=1.)
        self.guard_features=np.asarray(guard_features,int)
        self.audit.update(encoder_features=len(features),covariance_guard_features=len(guard_features),guard_panel_policy='Original384 past-selected genes; encoder expansion does not alter covariance acceptance panel.')

    def detection_probability(self,h,sl):
        return np.clip(self.pmean[sl]+h@self.detection[:,sl],1e-4,1-1e-4)

    def positive_log_change(self,h0,h1,sl,target):
        return (h1-h0)@self.positive_coef[:,sl]

    def positive_log_value(self,h1,sl,target):
        return self.positive_mean[sl]+h1@self.positive_coef[:,sl]-(self.positive_center[:,sl]*self.positive_coef[:,sl]).sum(0)

    def predict(self,target,mode,strength=1.,seed=20260928,sampling='independent'):
        if target<=self.cutoff or mode not in ['abundance','detection','joint'] or strength not in [.5,1.] or sampling not in ['independent','systematic']:raise ValueError('Invalid hurdle forecast')
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
                    change=self.positive_log_change(h0,h1,sl,target)
                    updated*=np.exp(np.clip(strength*backoff*change,-np.log(2.),np.log(2.)))
                if mode in ['detection','joint']:
                    p0=self.detection_probability(h0,sl)
                    p1=self.detection_probability(h1,sl)
                    change=np.clip(strength*backoff*(p1-p0),-.25,.25)*active
                    if sampling=='independent':
                        uniforms=rng.random(original.shape,dtype=np.float32)
                        added=(~positive)&(change>0)&(uniforms<change/(1-p0))
                        removed=positive&(change<0)&(uniforms<(-change)/p0)
                    else:
                        add_probability=np.where((~positive)&(change>0),np.clip(change/(1-p0),0,1),0)
                        remove_probability=np.where(positive&(change<0),np.clip((-change)/p0,0,1),0)
                        added=systematic_bernoulli(add_probability,rng)
                        removed=systematic_bernoulli(remove_probability,rng)
                    # Subtract each gene's positive-cell latent centroid for its conditional intercept.
                    log_positive=self.positive_log_value(h1,sl,target)
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
            change=covariance_change(self.donors[:,self.guard_features],result[:,self.guard_features])
            if np.isfinite(result).all() and change<=.4:break
        if not np.isfinite(result).all() or (result<0).any():raise ValueError('Invalid hurdle output')
        return result,np.arange(len(result)),{**self.audit,'mode':mode,'strength':strength,'backoff':backoff,
            'seed':seed,'sampling':sampling,'covariance_change_vs_reference':change,'newly_detected_entries':int(((self.donors==0)&(result>0)).sum()),
            'removed_detection_entries':int(((self.donors>0)&(result==0)).sum()),
            'abundance_factor_cap':2.,'detection_probability_delta_cap':.25,
            'common_uniform_policy':'Fixed seed reset across modes/strengths/backoffs; no seed selection. Systematic draws use random cell order and per-gene offsets, with dependent within-gene switches.',
            'mapped_mass_conserved':True,'protected_genes_unchanged':True}
