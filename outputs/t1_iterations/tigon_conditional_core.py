"""TIGON-inspired conditional-density core; unit endpoint masses, local solver/cost."""
import math
import torch
from torch import nn


def network(dim,width,layers,out):
    modules=[]
    for _ in range(layers):modules.extend([nn.Linear(dim,width),nn.Tanh()]);dim=width
    modules.append(nn.Linear(dim,out));return nn.Sequential(*modules)


class ConditionalUOT(nn.Module):
    def __init__(self,dim=8,width=16,growth=True):
        super().__init__();self.velocity=network(dim+1,width,4,dim);self.growth=network(dim+1,width,3,1);self.growth_enabled=growth

    def fields(self,t,z):
        inputs=torch.cat([torch.full((len(z),1),float(t),dtype=z.dtype,device=z.device),z],1)
        v=self.velocity(inputs);g=self.growth(inputs)
        return v,g if self.growth_enabled else g*0

    def forward(self,t,state):
        z,logw,logp,energy=state
        with torch.enable_grad():
            z=z.requires_grad_(True);v,g=self.fields(t,z)
            divergence=sum(torch.autograd.grad(v[:,i].sum(),z,create_graph=True,retain_graph=True)[0][:,i:i+1] for i in range(z.shape[1]))
            cost=(v.square().sum(1,keepdim=True)+g.square())*logw.exp()
        return v,g,g-divergence,cost


def integrate(field,z,left,right,step=.0625):
    if step<=0 or not math.isfinite(left+right):raise ValueError('Invalid time/step')
    zeros=z.new_zeros((len(z),1));state=(z,zeros,zeros,zeros)
    if left==right:return state
    steps=max(1,math.ceil(abs(right-left)/step));dt=(right-left)/steps
    for i in range(steps):
        first=field(left+i*dt,state)
        middle=tuple(a+.5*dt*b for a,b in zip(state,first))
        derivative=field(left+(i+.5)*dt,middle)
        state=tuple(a+dt*b for a,b in zip(state,derivative))
    return state


def log_density(values,centers,variance=.1,chunk=256):
    if variance<=0 or not len(centers):raise ValueError('Invalid kernel density')
    total=None
    constant=-.5*values.shape[1]*math.log(2*math.pi*variance)
    for start in range(0,len(centers),chunk):
        distances=(values[:,None,:]-centers[None,start:start+chunk,:]).square().sum(-1)
        block=torch.logsumexp(constant-distances/(2*variance),dim=1)
        total=block if total is None else torch.logaddexp(total,block)
    return total-math.log(len(centers))


def objective(model,groups,times,queries,initial,variance=.1):
    # All endpoint masses are ONE; captured counts never multiply the density.
    loss=initial.new_zeros(())
    for i in range(1,len(times)):
        query=queries[i-1];target=log_density(query,groups[i],variance).exp()
        for j in [0,i-1]:
            z,_,delta,_=integrate(model,query,times[i],times[j])
            predicted=(log_density(z,groups[j],variance)-delta[:,0]).exp()
            loss=loss+1e4*(target-predicted).square().mean()
    _,_,_,energy=integrate(model,initial,times[0],times[-1])
    return loss+(times[-1]-times[0])*energy.mean()


def conditional_weights(logw):
    weights=torch.softmax(logw.flatten(),0)
    return weights,1/weights.square().sum()
