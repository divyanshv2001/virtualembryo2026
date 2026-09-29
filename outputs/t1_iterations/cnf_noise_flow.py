"""Past-only midpoint manifold-density regularization for continuous flows."""
import math
import numpy as np
import torch
from cnf_density_flow import DensityFlowNet,inverse_density


def rk4_position(field,z,start,finish,step=.0625):
    count=max(1,int(math.ceil(abs(finish-start)/step)));dt=(finish-start)/count
    for i in range(count):
        t=start+i*dt;a=field(t,z);b=field(t+dt/2,z+dt*a/2);c=field(t+dt/2,z+dt*b/2);d=field(t+dt,z+dt*c)
        z=z+dt*(a+2*b+2*c+d)/6
    return z


def manifold_penalty(points,reference):
    if len(reference)<5:raise ValueError('Five reference cells required')
    # Author-style five-neighbor hinge; reference contains past cells only.
    neighbors=torch.cdist(points,reference).topk(5,dim=1,largest=False).values
    return torch.clamp(neighbors-.1,min=0.).mean()


def train_noise_density(net,z,stages,energy_weight,checkpoint,emit,resume=False,steps=400,density_weight=0.,observation_noise=0.):
    if observation_noise not in [0.,.05,.1]:raise ValueError('Undeclared observation noise')
    noise_rng=torch.Generator().manual_seed(20260929)
    if density_weight not in [0.,1.,10.]:raise ValueError('Undeclared density weight')
    if energy_weight not in [0.,.1]:raise ValueError('Undeclared energy weight')
    torch.manual_seed(20260928);optimizer=torch.optim.Adam(net.parameters(),lr=.001);start=0;history=[]
    if resume and checkpoint.exists():
        saved=torch.load(checkpoint,weights_only=False,map_location='cpu')
        if saved['energy_weight']!=energy_weight or saved['steps']!=steps or saved['density_weight']!=density_weight or saved['observation_noise']!=observation_noise:raise ValueError('Checkpoint configuration mismatch')
        noise_rng.set_state(saved['observation_rng']);net.load_state_dict(saved['net']);optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng']);start=saved['step'];history=saved['history']
    groups=[torch.tensor(z[stages==t],dtype=torch.float32) for t in np.unique(stages)];times=np.unique(stages)-net.origin
    reference=torch.tensor(z,dtype=torch.float32)
    for step in range(start,steps):
        group=int(torch.randint(len(groups),(1,)));source=groups[group][torch.randint(len(groups[group]),(64,))].clone()
        if observation_noise:source=source+observation_noise*torch.randn(source.shape,generator=noise_rng)
        noise=torch.randint(0,2,source.shape).float()*2-1
        nll,energy=inverse_density(net.velocity,source,float(times[group]),noise=noise)
        density=torch.tensor(0.)
        if density_weight and group>0:
            midpoint=rk4_position(net.velocity,source,float(times[group]),float(times[group])-.125)
            density=manifold_penalty(midpoint,reference)
        loss=nll.mean()+energy_weight*energy.mean()+density_weight*density
        if not torch.isfinite(loss):raise ValueError('Nonfinite density loss')
        optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
        if not torch.isfinite(norm):raise ValueError('Nonfinite density gradient')
        optimizer.step()
        if (step+1)%50==0 or step+1==steps:
            record={'step':step+1,'loss':float(loss.detach()),'negative_log_likelihood':float(nll.mean().detach()),'energy':float(energy.mean().detach()),'manifold_density':float(density.detach()),'sampled_past_stage':float(times[group]+net.origin)};history.append(record)
            torch.save({'net':net.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state(),'step':step+1,'steps':steps,'energy_weight':energy_weight,'density_weight':density_weight,'observation_noise':observation_noise,'observation_rng':noise_rng.get_state(),'history':history},checkpoint)
            emit('cnf_training_checkpoint',**record)
    net.eval();return history
