"""Past adjacent-snapshot MMD fine-tuning of a frozen-initialized CNF."""
import numpy as np
import torch
from cnf_density_flow import inverse_density
from cnf_manifold_flow import rk4_position,manifold_penalty


def kernel_mmd(x,y,bandwidth):
    if bandwidth<=0:raise ValueError('Positive bandwidth required')
    scales=x.new_tensor([.25,.5,1.,2.,4.])*bandwidth
    def kernel(a,b):
        distances=torch.cdist(a,b).square()
        return torch.exp(-distances[:,:,None]/(2*scales)).mean(2)
    # Biased empirical squared MMD is a nonnegative optimization surrogate.
    return kernel(x,x).mean()+kernel(y,y).mean()-2*kernel(x,y).mean()


def train_transition_mmd(net,z,stages,initial,checkpoint,emit,weight=1.,updates=200,resume=False):
    if weight not in [0.,1.,10.] or updates not in [2,200]:raise ValueError('Undeclared transition ablation')
    optimizer=torch.optim.Adam(net.parameters(),lr=.001)
    saved=torch.load(checkpoint if resume and checkpoint.exists() else initial,weights_only=False,map_location='cpu')
    if saved['energy_weight']!=.1 or saved['density_weight']!=10.:raise ValueError('Initial learner mismatch')
    if 'transition_mmd_weight' in saved and (saved['transition_mmd_weight']!=weight or saved['updates']!=updates):raise ValueError('Resume configuration mismatch')
    net.load_state_dict(saved['net']);optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
    generator=torch.Generator().manual_seed(20260930)
    if 'mmd_rng' in saved:generator.set_state(saved['mmd_rng'])
    initial_step=saved.get('initial_step',saved['step']);start=saved['step'];finish=initial_step+updates
    history=saved['history'].copy();unique=np.unique(stages);times=unique-net.origin
    if unique.max()>net.cutoff or len(unique)<2:raise ValueError('Invalid past transition stages')
    groups=[torch.tensor(z[stages==t],dtype=torch.float32) for t in unique];reference=torch.tensor(z,dtype=torch.float32)
    sample=np.asarray(z)[np.random.default_rng(20260930).choice(len(z),min(512,len(z)),replace=False)]
    distances=np.square(sample[:,None,:]-sample[None,:,:]).sum(2);bandwidth=float(np.maximum(np.median(distances[np.triu_indices(len(sample),1)]),1e-4))
    for step in range(start,finish):
        group=int(torch.randint(len(groups),(1,)));source=groups[group][torch.randint(len(groups[group]),(64,))].clone();noise=torch.randint(0,2,source.shape).float()*2-1
        nll,energy=inverse_density(net.velocity,source,float(times[group]),noise=noise)
        density=source.new_zeros(())
        if group>0:density=manifold_penalty(rk4_position(net.velocity,source,float(times[group]),float(times[group])-.125),reference)
        loss=nll.mean()+.1*energy.mean()+10.*density;mmd=source.new_zeros(())
        if weight:
            interval=int(torch.randint(len(groups)-1,(1,),generator=generator))
            a=groups[interval][torch.randint(len(groups[interval]),(64,),generator=generator)]
            b=groups[interval+1][torch.randint(len(groups[interval+1]),(64,),generator=generator)]
            transported=rk4_position(net.velocity,a,float(times[interval]),float(times[interval+1]),step=.125)
            mmd=kernel_mmd(transported,b,bandwidth);loss=loss+weight*mmd
        if not torch.isfinite(loss):raise ValueError('Nonfinite transition objective')
        optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
        if not torch.isfinite(norm):raise ValueError('Nonfinite transition gradient')
        optimizer.step()
        if (step+1)%50==0 or step+1==finish:
            record={'step':step+1,'loss':float(loss.detach()),'negative_log_likelihood':float(nll.mean().detach()),'energy':float(energy.mean().detach()),'manifold_density':float(density.detach()),'transition_mmd':float(mmd.detach()),'transition_mmd_weight':weight,'past_bandwidth_squared':bandwidth,'sampled_past_stage':float(unique[group]),'transition_source_stage':float(unique[interval]) if weight else None,'transition_target_stage':float(unique[interval+1]) if weight else None};history.append(record)
            torch.save({'net':net.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state(),'mmd_rng':generator.get_state(),'step':step+1,'steps':finish,'initial_step':initial_step,'updates':updates,'energy_weight':.1,'density_weight':10.,'transition_mmd_weight':weight,'history':history},checkpoint)
            emit('cnf_training_checkpoint',**record)
    net.eval();return history
