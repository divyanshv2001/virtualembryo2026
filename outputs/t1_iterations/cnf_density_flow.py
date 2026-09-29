"""CPU likelihood/energy continuous-flow adaptation on past PCA coordinates."""
import math
import numpy as np
import torch
from torch import nn


def divergence(field,z,noise=None):
    if noise is None:
        return sum(torch.autograd.grad(field[:,j].sum(),z,create_graph=True,retain_graph=True)[0][:,j] for j in range(z.shape[1]))
    return (torch.autograd.grad((field*noise).sum(),z,create_graph=True,retain_graph=True)[0]*noise).sum(1)


def inverse_density(field,values,time,step=.125,noise=None,log_base=None):
    z=values.requires_grad_(True);delta=torch.zeros(len(z));energy=torch.zeros(len(z))
    count=max(1,int(math.ceil(abs(time)/step)));dt=-time/count
    def rhs(t,state):
        position=state[0];velocity=field(t,position)
        return velocity,-divergence(velocity,position,noise),velocity.square().sum(1)
    for i in range(count):
        t=time+i*dt;s=(z,delta,energy);k1=rhs(t,s)
        k2=rhs(t+dt/2,tuple(a+dt*b/2 for a,b in zip(s,k1)))
        k3=rhs(t+dt/2,tuple(a+dt*b/2 for a,b in zip(s,k2)))
        k4=rhs(t+dt,tuple(a+dt*b for a,b in zip(s,k3)))
        z,delta,energy=tuple(a+dt*(b+2*c+2*d+e)/6 for a,b,c,d,e in zip(s,k1,k2,k3,k4))
    logbase=-.5*(z.square()+math.log(2*math.pi)).sum(1) if log_base is None else log_base(z)
    return -logbase+delta,-energy


class DensityFlowNet(nn.Module):
    def __init__(self,basis,pca_center,cutoff,origin,width=64):
        super().__init__();self.cutoff=cutoff;self.origin=origin
        if width not in [64,128,256]:raise ValueError('Undeclared flow width')
        self.width=width
        self.register_buffer('basis',torch.tensor(basis,dtype=torch.float32))
        self.register_buffer('pca_center',torch.tensor(pca_center,dtype=torch.float32))
        d=len(basis);self.field=nn.Sequential(nn.Linear(d+1,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,d))
        nn.init.zeros_(self.field[-1].weight);nn.init.zeros_(self.field[-1].bias)
    def encode(self,values):
        z=(values-self.pca_center)@self.basis.T;return z,torch.zeros_like(z)
    def velocity(self,time,z):
        return self.field(torch.cat([z,torch.full((len(z),1),float(time),dtype=z.dtype)],1))
    def trajectory(self,z,times,step=.125):
        history=[z]
        for previous,target in zip(times[:-1],times[1:]):
            span=float(target-previous);n=max(1,int(math.ceil(abs(span)/step)));dt=span/n
            for j in range(n):
                t=self.cutoff-self.origin+float(previous)+j*dt
                a=self.velocity(t,z);b=self.velocity(t+dt/2,z+dt*a/2);c=self.velocity(t+dt/2,z+dt*b/2);d=self.velocity(t+dt,z+dt*c)
                z=z+dt*(a+2*b+2*c+d)/6
            history.append(z)
        return torch.stack(history)


def train_density(net,z,stages,energy_weight,checkpoint,emit,resume=False,steps=400):
    if energy_weight not in [0.,.1]:raise ValueError('Undeclared energy weight')
    torch.manual_seed(20260928);optimizer=torch.optim.Adam(net.parameters(),lr=.001);start=0;history=[]
    if resume and checkpoint.exists():
        saved=torch.load(checkpoint,weights_only=False,map_location='cpu')
        if saved['energy_weight']!=energy_weight or saved['steps']!=steps:raise ValueError('Checkpoint configuration mismatch')
        net.load_state_dict(saved['net']);optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng']);start=saved['step'];history=saved['history']
    groups=[torch.tensor(z[stages==t],dtype=torch.float32) for t in np.unique(stages)];times=np.unique(stages)-net.origin
    for step in range(start,steps):
        group=int(torch.randint(len(groups),(1,)));source=groups[group][torch.randint(len(groups[group]),(64,))].clone()
        noise=torch.randint(0,2,source.shape).float()*2-1
        nll,energy=inverse_density(net.velocity,source,float(times[group]),noise=noise)
        loss=nll.mean()+energy_weight*energy.mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite density loss')
        optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
        if not torch.isfinite(norm):raise ValueError('Nonfinite density gradient')
        optimizer.step()
        if (step+1)%50==0 or step+1==steps:
            record={'step':step+1,'loss':float(loss.detach()),'negative_log_likelihood':float(nll.mean().detach()),'energy':float(energy.mean().detach()),'sampled_past_stage':float(times[group]+net.origin)};history.append(record)
            torch.save({'net':net.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state(),'step':step+1,'steps':steps,'energy_weight':energy_weight,'history':history},checkpoint)
            emit('cnf_training_checkpoint',**record)
    net.eval();return history
