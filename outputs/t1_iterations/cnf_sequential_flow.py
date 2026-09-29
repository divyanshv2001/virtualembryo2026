"""Shared backward multi-stage CNF likelihood; past stages only."""
import math
import numpy as np
import torch
from cnf_density_flow import DensityFlowNet,divergence
from cnf_manifold_flow import manifold_penalty,rk4_position


def backward_interval(field,z,start,finish,noise=None,step=.125):
    z=z.requires_grad_(True);delta=z.new_zeros(len(z));energy=z.new_zeros(len(z))
    count=max(1,int(math.ceil(abs(finish-start)/step)));dt=(finish-start)/count
    def rhs(t,state):
        velocity=field(t,state[0])
        return velocity,-divergence(velocity,state[0],noise),velocity.square().sum(1)
    for i in range(count):
        t=start+i*dt;s=(z,delta,energy);a=rhs(t,s)
        b=rhs(t+dt/2,tuple(v+dt*k/2 for v,k in zip(s,a)))
        c=rhs(t+dt/2,tuple(v+dt*k/2 for v,k in zip(s,b)))
        d=rhs(t+dt,tuple(v+dt*k for v,k in zip(s,c)))
        z,delta,energy=tuple(v+dt*(k+2*l+2*m+n)/6 for v,k,l,m,n in zip(s,a,b,c,d))
    return z,delta,-energy


def sequential_terms(field,samples,times,noises=None):
    if len(samples)!=len(times) or not len(times) or np.any(np.diff(times)<=0) or times[0]<=0:raise ValueError('Increasing positive past times required')
    position=None;delta=None;energy=None;trace=None;order=[]
    for i in range(len(times)-1,-1,-1):
        source=samples[i].clone();n=len(source)
        if position is None:
            position=source;delta=source.new_zeros(n);energy=source.new_zeros(n)
            trace=None if noises is None else noises[i]
        else:
            position=torch.cat([position,source]);delta=torch.cat([delta,source.new_zeros(n)]);energy=torch.cat([energy,source.new_zeros(n)])
            if noises is not None:trace=torch.cat([trace,noises[i]])
        order.append((i,n))
        position,change,action=backward_interval(field,position,float(times[i]),float(times[i-1]) if i else 0.,trace)
        delta=delta+change;energy=energy+action
    likelihood=.5*(position.square()+math.log(2*math.pi)).sum(1)+delta
    nll=[None]*len(times);action=[None]*len(times);offset=0
    for index,count in order:
        nll[index]=likelihood[offset:offset+count].mean();action[index]=energy[offset:offset+count].mean();offset+=count
    return torch.stack(nll).mean(),torch.stack(action).mean()


def train_sequential_density(net,z,stages,energy_weight,checkpoint,emit,resume=False,steps=800,density_weight=10.):
    if steps not in [2,4,400,800] or energy_weight!=.1 or density_weight!=10.:raise ValueError('Undeclared sequential configuration')
    torch.manual_seed(20260928);optimizer=torch.optim.Adam(net.parameters(),lr=.001);start=0;history=[]
    if resume and checkpoint.exists():
        saved=torch.load(checkpoint,weights_only=False,map_location='cpu')
        if saved['steps']!=steps or saved['energy_weight']!=energy_weight or saved['density_weight']!=density_weight:raise ValueError('Checkpoint configuration mismatch')
        net.load_state_dict(saved['net']);optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng']);start=saved['step'];history=saved['history']
    unique=np.unique(stages);times=unique-net.origin
    groups=[torch.tensor(z[stages==t],dtype=torch.float32) for t in unique];reference=torch.tensor(z,dtype=torch.float32)
    if len(groups)>64 or unique.max()>net.cutoff:raise ValueError('Invalid permitted-stage coverage')
    sizes=[64//len(groups)+(i<64%len(groups)) for i in range(len(groups))]
    for update in range(start,steps):
        samples=[g[torch.randint(len(g),(n,))].clone() for g,n in zip(groups,sizes)]
        noises=[torch.randint(0,2,s.shape).float()*2-1 for s in samples]
        nll,energy=sequential_terms(net.velocity,samples,times,noises)
        penalties=[manifold_penalty(rk4_position(net.velocity,s,float(t),float(t)-.125),reference) for s,t in zip(samples[1:],times[1:])]
        # Equal-stage expectation includes earliest stage's zero manifold term.
        density=torch.stack(penalties).sum()/len(groups) if penalties else nll.new_zeros(())
        loss=nll+energy_weight*energy+density_weight*density
        if not torch.isfinite(loss):raise ValueError('Nonfinite sequential objective')
        optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
        if not torch.isfinite(norm):raise ValueError('Nonfinite sequential gradient')
        optimizer.step()
        if (update+1)%50==0 or update+1==steps:
            record={'step':update+1,'loss':float(loss.detach()),'negative_log_likelihood':float(nll.detach()),'energy':float(energy.detach()),'manifold_density':float(density.detach()),'all_past_stages':unique.tolist(),'per_stage_batch_sizes':sizes};history.append(record)
            torch.save({'net':net.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state(),'step':update+1,'steps':steps,'energy_weight':energy_weight,'density_weight':density_weight,'history':history},checkpoint)
            emit('cnf_training_checkpoint',**record)
    net.eval();return history
