"""Local scNODE-inspired joint representation/dynamics; not author reproduction.

Protocol reference: rsinghlab/scNODE commit ba55e8ddecfb74874047517952ab799326b4126d.
Bounded minibatches replace author whole-data MSE pretraining. No pretrained weights.
"""
import torch
from torch import nn
from torchdiffeq import odeint
from geomloss import SamplesLoss


class JointModel(nn.Module):
    def __init__(self, genes, latent=32, width=128):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(genes,width),nn.Identity(),nn.Linear(width,latent))
        self.mu=nn.Linear(latent,latent)
        self.std=nn.Linear(latent,latent)
        self.decoder=nn.Sequential(nn.Linear(latent,width),nn.ReLU(),nn.Linear(width,genes),nn.ReLU())
        self.drift=nn.Sequential(nn.Linear(latent,64),nn.ReLU(),nn.Linear(64,latent),nn.ReLU())

    def encode(self,x):
        h=self.encoder(x)
        return self.mu(h), self.std(h).abs().clamp_min(1e-5)

    def sample(self,x,noise):
        mu,std=self.encode(x)
        return mu+std*noise

    def trajectory(self,z,times):
        return odeint(lambda t,y:self.drift(y),z,times,method='euler',rtol=1e-5,atol=1e-5).transpose(0,1)


def joint_loss(model,batches,times,noises,beta,targets=None):
    if beta not in [0.,.1]: raise ValueError('Undeclared dynamic regularization')
    latent=model.trajectory(model.sample(batches[0],noises[0]),times)
    decoded=model.decoder(latent)
    sinkhorn=SamplesLoss('sinkhorn',p=2,blur=.05,scaling=.5,debias=True,backend='tensorized')
    expression=sum(sinkhorn(x,decoded[:,i]) for i,x in enumerate(batches if targets is None else targets))/len(batches)
    # Computed in both arms to keep draw order and computation comparable.
    dynamic=sum(sinkhorn(model.sample(x,noises[i+1]),latent[:,i]) for i,x in enumerate(batches))/len(batches)
    return expression+beta*dynamic,expression,dynamic
