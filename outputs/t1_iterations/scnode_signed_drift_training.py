"""Isolated signed final-drift activation ablation, fresh cutoff and shared draws."""
import copy,hashlib
import numpy as np
import torch
from scnode_past_fold_training import JointModel as SubstepModel
from scnode_joint_model import joint_loss
from scnode_biological import batch

class JointModel(SubstepModel):
    def __init__(self,genes,signed=False):
        super().__init__(genes)
        if signed:self.drift[-1]=torch.nn.Identity()

def train_pair(x,stages,columns,cutoff,run,emit):
    eligible=np.flatnonzero(stages<=cutoff)
    if not len(eligible): raise ValueError('No past fit rows')
    times=np.unique(stages[eligible]);groups=[np.flatnonzero(stages==t) for t in times]
    if times.max()>cutoff: raise ValueError('Future fit rows')
    torch.manual_seed(20261002);rng=np.random.default_rng(20261002)
    model=JointModel(len(columns))
    optimizer=torch.optim.Adam(list(model.encoder.parameters())+list(model.mu.parameters())+list(model.std.parameters())+list(model.decoder.parameters()),lr=.001,betas=(.95,.99))
    for step in range(200):
        group=groups[int(rng.integers(len(groups)))];values=batch(x,rng.choice(group,64,replace=True),columns)
        latent=model.sample(values,torch.randn(64,32));loss=((model.decoder(latent)-values)**2).mean()
        if not torch.isfinite(loss): raise ValueError('Nonfinite biological pretrain')
        optimizer.zero_grad();loss.backward();optimizer.step()
        if (step+1)%50==0: emit('vae_pretraining',step=step+1,loss=float(loss.detach()))
    initial=copy.deepcopy(model.state_dict());torch.save(initial,run/'shared_pretrain.pt')
    flows={};matched_stream=None
    for signed in [False,True]:
        beta=.1
        name='linear_drift_beta_0.1' if signed else 'relu_drift_beta_0.1'
        model=JointModel(len(columns),signed=signed)
        model.load_state_dict(initial);torch.manual_seed(20261002);rng=np.random.default_rng(20261002)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001,betas=(.95,.99));history=[];stream=hashlib.sha256()
        for step in range(1000):
            batch_rows=[rng.choice(g,32,replace=True) for g in groups]
            target_rows=[rng.choice(g,200,replace=True) for g in groups]
            for rows in batch_rows+target_rows:stream.update(rows.tobytes())
            batches=[batch(x,rows,columns) for rows in batch_rows]
            targets=[batch(x,rows,columns) for rows in target_rows]
            noises=[torch.randn(32,32) for _ in range(len(groups)+1)]
            for noise in noises:stream.update(noise.numpy().tobytes())
            loss,expression,dynamic=joint_loss(model,batches,torch.tensor(times-times[0],dtype=torch.float32),noises,beta,targets=targets)
            if not torch.isfinite(loss): raise ValueError('Nonfinite biological joint loss')
            optimizer.zero_grad();loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()): raise ValueError('Nonfinite biological gradient')
            optimizer.step()
            if (step+1)%25==0:
                record={'candidate':name,'step':step+1,'loss':float(loss.detach()),'expression_loss':float(expression.detach()),'dynamic_loss':float(dynamic.detach())};history.append(record)
                torch.save({'net':model.state_dict(),'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'numpy_rng':rng.bit_generator.state,'history':history,'beta':beta,'signed_drift':signed,'fit_max_stage':float(times.max()),'seed':20261002,'steps':1000},run/(name+'.pt'));emit('scnode_training_checkpoint',**record)
        if matched_stream is not None and stream.hexdigest()!=matched_stream: raise ValueError('New arm random streams differ')
        matched_stream=stream.hexdigest();emit('matched_training_stream',candidate=name,draw_sha256=matched_stream)
        flows[name]=copy.deepcopy(model).eval()
    return flows

