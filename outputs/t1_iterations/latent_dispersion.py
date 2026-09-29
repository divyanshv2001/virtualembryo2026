"""Fixed terminal latent dispersion sensitivity; not a learned diffusion model."""
import numpy as np
import torch


class LatentDispersion:
    def __init__(self,base,scale=0.,seed=2026092917):
        self.base=base;self.cutoff=base.cutoff;self.origin=base.origin
        self.configure(scale,seed)
    def configure(self,scale,seed=2026092917):
        if scale not in [0.,.025,.05,.1]:raise ValueError('Undeclared dispersion scale')
        self.scale=scale;self.seed=seed
    def encode(self,values):return self.base.encode(values)
    def trajectory(self,z,times):
        result=self.base.trajectory(z,times)
        if self.scale==0.:return result
        result=result.clone();n,d=z.shape;rng=np.random.default_rng(self.seed)
        # Antithetic paired rows avoid accidental global mean drift. Odd final row=0.
        draw=rng.standard_normal((n//2,d));draw=np.concatenate([draw,-draw],axis=0)
        if n%2:draw=np.concatenate([draw,np.zeros((1,d))],axis=0)
        draw=draw[rng.permutation(n)]
        noise=torch.as_tensor(draw,dtype=z.dtype,device=z.device)
        for i in range(1,len(times)):
            elapsed=float(times[i]-times[0])
            if elapsed<0:raise ValueError('Backward dispersion undeclared')
            result[i]+=self.scale*np.sqrt(elapsed)*noise
        return result
