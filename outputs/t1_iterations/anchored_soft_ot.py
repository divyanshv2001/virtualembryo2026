"""Bounded likelihood-anchored past-only soft transport ablation."""
import numpy as np
import torch
from cnf_manifold_flow import inverse_density, rk4_position, manifold_penalty
from soft_ot_flow_matching import soft_coupling


def combine_gradients(base, auxiliary, weight=.01, cap=.1):
    base_norm = torch.sqrt(sum((g*g).sum() for g in base))
    auxiliary_norm = torch.sqrt(sum((g*g).sum() for g in auxiliary))
    factor = min(weight, cap*float(base_norm)/max(float(auxiliary_norm), 1e-30))
    if float(base_norm) == 0:
        factor = 0.
    return [b+factor*a for b,a in zip(base,auxiliary)], factor, float(base_norm), float(auxiliary_norm)



def project_auxiliary(base, auxiliary):
    """Remove only the component opposing the past likelihood objective."""
    squared=sum((b*b).sum() for b in base)
    dot=sum((b*a).sum() for b,a in zip(base,auxiliary))
    coefficient=min(float(dot)/max(float(squared),1e-30),0.) if float(squared)>0 else 0.
    return [a-coefficient*b for b,a in zip(base,auxiliary)]

def train_anchored(net, z, stages, cutoff, checkpoint, emit, steps=400, batch_size=64, seed=20260928):
    z, stages = np.asarray(z,dtype=np.float32), np.asarray(stages,dtype=float)
    if stages.max()>cutoff or not np.isfinite(z).all() or not np.isfinite(stages).all():
        raise ValueError('Invalid past-only training data')
    if steps!=400 or batch_size!=64 or seed!=20260928:
        raise ValueError('Outside declared ablation')
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    parameters=list(net.parameters())
    initial=[p.detach().clone() for p in parameters]
    radius=.05*torch.sqrt(sum((p*p).sum() for p in initial))
    optimizer=torch.optim.Adam(parameters,lr=.001)
    absolute=np.unique(stages)
    times=absolute-net.origin
    groups=[torch.tensor(z[stages==t],dtype=torch.float32) for t in absolute]
    reference=torch.tensor(z,dtype=torch.float32)
    history=[]
    for step in range(steps):
        # Exact original NLL Torch draw order; all OT randomness is NumPy.
        group=int(torch.randint(len(groups),(1,)))
        source=groups[group][torch.randint(len(groups[group]),(batch_size,))].clone()
        noise=torch.randint(0,2,source.shape).float()*2-1
        nll,energy=inverse_density(net.velocity,source,float(times[group]),noise=noise,log_base=getattr(net,'log_base',None))
        density=torch.tensor(0.)
        if group>0:
            midpoint=rk4_position(net.velocity,source,float(times[group]),float(times[group])-.125)
            density=manifold_penalty(midpoint,reference)
        loss=nll.mean()+.1*energy.mean()+10.*density
        base=[torch.zeros_like(p) if g is None else g for p,g in
              zip(parameters,torch.autograd.grad(loss,parameters,allow_unused=True))]
        pair=int(rng.integers(len(times)-1))
        left=groups[pair][rng.choice(len(groups[pair]),batch_size,replace=True)].numpy()
        right=groups[pair+1][rng.choice(len(groups[pair+1]),batch_size,replace=True)].numpy()
        coupling,audit=soft_coupling(left,right)
        sampled=rng.choice(coupling.size,batch_size,replace=True,p=coupling.ravel())
        start=torch.from_numpy(left[sampled//len(right)])
        finish=torch.from_numpy(right[sampled%len(right)])
        fraction=torch.tensor(rng.random((batch_size,1)),dtype=torch.float32)
        elapsed=float(times[pair+1]-times[pair])
        point=start+fraction*(finish-start)
        clock=float(times[pair])+fraction*elapsed
        auxiliary_loss=((net.field(torch.cat([point,clock],dim=1))-(finish-start)/elapsed)**2).mean()
        auxiliary=torch.autograd.grad(auxiliary_loss,parameters)
        gradients,factor,base_norm,aux_norm=combine_gradients(base,auxiliary)
        if not all(torch.isfinite(g).all() for g in gradients) or not torch.isfinite(loss) or not torch.isfinite(auxiliary_loss):
            raise ValueError('Nonfinite anchored gradient/loss')
        optimizer.zero_grad()
        for p,g in zip(parameters,gradients): p.grad=g
        torch.nn.utils.clip_grad_norm_(parameters,5.)
        optimizer.step()
        with torch.no_grad():
            displacement=torch.sqrt(sum(((p-i)**2).sum() for p,i in zip(parameters,initial)))
            projected=bool(displacement>radius)
            if projected:
                for p,i in zip(parameters,initial): p.copy_(i+(p-i)*radius/displacement)
            relative=float(torch.sqrt(sum(((p-i)**2).sum() for p,i in zip(parameters,initial)))/(radius/.05))
        if relative>.050001 or factor*aux_norm>base_norm*.100001+1e-8:
            raise ValueError('Declared bound violated')
        record={'step':step+1,'loss':float(loss.detach()),'ot_loss':float(auxiliary_loss.detach()),
                'effective_ot_weight':factor,'nll_gradient_norm':base_norm,'ot_gradient_norm':aux_norm,
                'relative_parameter_distance':relative,'projected':projected,**audit}
        history.append(record)
        if (step+1)%50==0:
            torch.save({'net':net.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state(),
                        'numpy_rng':rng.bit_generator.state,'history':history,'fit_max_stage':float(stages.max()),
                        'seed':seed,'steps':steps,'batch_size':batch_size},checkpoint)
            emit('anchored_training_checkpoint',**record)
    net.eval()
    return history
