"""FLeCS-inspired predictive kinetics in the fixed PCA geometry; not reproduction.

Candidate adult binding edges do not identify signed causal regulation. The
reconstructed nonnegative gene state is approximate; output heads stay fixed.
"""
import copy
import math
import time
import numpy as np
import torch
from geomloss import SamplesLoss
from cnf_manifold_flow import rk4_position


def shuffled_edges(source, target, n_genes, seed=20261005):
    """Relabel targets within exact indegree classes; node degrees stay fixed.

    This bounded identity control is not a uniform draw from all degree-matched
    graphs. Singleton classes and structurally identical columns can persist.
    """
    degree=np.bincount(target,minlength=n_genes)
    permutation=np.arange(n_genes);rng=np.random.default_rng(seed)
    for d in np.unique(degree):
        group=np.flatnonzero(degree==d);permutation[group]=rng.permutation(group)
    shuffled=permutation[target]
    original=set(zip(source.tolist(),target.tolist()))
    overlap=len(original.intersection(zip(source.tolist(),shuffled.tolist())))
    changed=len(source)-overlap
    if changed==0:raise ValueError('Graph identity control unchanged')
    return source.copy(),shuffled,{'strategy':'target_labels_within_exact_indegree_classes','changed_edges':changed,'shared_edges':overlap,'changed_fraction':changed/len(source),'permuted_target_labels':int(np.count_nonzero(permutation!=np.arange(n_genes))),'limitation':'Not uniform degree-matched graph sampling; singleton degrees and common edges persist.'}


class KineticFlow(torch.nn.Module):
    def __init__(self, drift, gene_mean, source, target, kind):
        super().__init__()
        if kind not in ('real','shuffle','none'): raise ValueError('Unknown kinetic graph')
        self.drift=copy.deepcopy(drift).eval();self.kind=kind
        self.cutoff,self.origin=drift.cutoff,drift.origin
        for p in self.drift.parameters():p.requires_grad_(False)
        n=drift.basis.shape[1]
        if len(gene_mean)!=n or (np.asarray(gene_mean)<0).any():raise ValueError('Invalid past gene scale')
        self.register_buffer('inverse',torch.linalg.pinv(drift.basis.T))
        self.register_buffer('gene_mean',torch.tensor(gene_mean,dtype=torch.float32))
        self.register_buffer('amplitude',.2*torch.tensor(gene_mean,dtype=torch.float32))
        self.register_buffer('edges',torch.tensor(np.stack([target,source]),dtype=torch.long))
        torch.sparse_coo_tensor(self.edges,torch.ones(len(source)),(n,n),check_invariants=True)
        degree=np.maximum(np.bincount(target,minlength=n),1)
        self.register_buffer('edge_scale',torch.tensor(1/np.sqrt(degree[target]),dtype=torch.float32))
        self.edge_weight=torch.nn.Parameter(torch.zeros(len(source)))
        self.bias=torch.nn.Parameter(torch.zeros(n))
        self.alpha=torch.nn.Parameter(torch.full((n,),math.log(math.expm1(.1))))
        self.register_buffer('initial_alpha',torch.nn.functional.softplus(self.alpha.detach()).clone())

    def encode(self,x):return self.drift.encode(x)

    def velocity(self,t,z):
        q=torch.clamp(self.gene_mean+z@self.inverse,min=0.)
        if self.edge_weight.numel():
            adjacency=torch.sparse_coo_tensor(self.edges,torch.tanh(self.edge_weight)*self.edge_scale,(len(self.bias),len(self.bias)),check_invariants=False)
            messages=torch.sparse.mm(adjacency,(q-self.gene_mean).T).T
        else:messages=torch.zeros_like(q)
        production_delta=self.amplitude*(torch.sigmoid(messages+self.bias)-.5)
        decay_delta=(torch.nn.functional.softplus(self.alpha)-self.initial_alpha)*q
        residual=(production_delta-decay_delta)@self.drift.basis.T
        residual=residual/torch.clamp(torch.linalg.vector_norm(residual,dim=1,keepdim=True),min=1.)
        return self.drift.velocity(t,z)+residual

    def trajectory(self,z,times,step=.125):
        history=[z]
        for a,b in zip(times[:-1],times[1:]):
            z=rk4_position(self.velocity,z,self.cutoff-self.origin+float(a),self.cutoff-self.origin+float(b),step)
            history.append(z)
        return torch.stack(history)


def train_kinetics(model,coordinates,stages,checkpoint,emit,steps=400,batch=64):
    times=np.unique(stages)
    if len(times)<3 or times.max()>model.cutoff:raise ValueError('Past-only stage support')
    groups=[torch.tensor(coordinates[stages==t],dtype=torch.float32) for t in times]
    if min(map(len,groups))<batch:raise ValueError('Small permitted stage')
    rng=torch.Generator().manual_seed(20261004)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
    loss_fn=SamplesLoss('sinkhorn',p=2,blur=.05,scaling=.9,backend='tensorized')
    frozen={k:v.clone() for k,v in model.drift.state_dict().items()};history=[]
    for i in range(steps):
        j=int(torch.randint(len(groups)-1,(1,),generator=rng))
        a=groups[j][torch.randint(len(groups[j]),(batch,),generator=rng)]
        b=groups[j+1][torch.randint(len(groups[j+1]),(batch,),generator=rng)]
        prediction=rk4_position(model.velocity,a,float(times[j]-model.origin),float(times[j+1]-model.origin),step=.125)
        loss=loss_fn(prediction,b)
        if not torch.isfinite(loss):raise ValueError('Nonfinite kinetic loss')
        optimizer.zero_grad();loss.backward()
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.)
        if not torch.isfinite(norm):raise ValueError('Nonfinite kinetic gradient')
        optimizer.step()
        if (i+1)%50==0 or i+1==steps:
            record={'step':i+1,'loss':float(loss.detach()),'source_stage':float(times[j]),'destination_stage':float(times[j+1])}
            history.append(record);emit('kinetic_training_checkpoint',graph_kind=model.kind,**record)
    for k,v in model.drift.state_dict().items():torch.testing.assert_close(v,frozen[k],rtol=0,atol=0)
    torch.save({'net':model.state_dict(),'kind':model.kind,'steps':steps,'batch_size':batch,'fit_max_stage':float(times.max()),'history':history,'frozen_drift_exact':True},checkpoint)
    model.eval();return history


def prepare_kinetic_models(initial,c,run,emit):
    from pathlib import Path
    from scipy import sparse
    from run_t1 import digest
    folder=Path(__file__).resolve().parent/'private/flecs_adult_graph_01'
    graph=sparse.load_npz(folder/'past4096_adult_graph.npz').tocoo()
    if digest(folder/'past4096_adult_graph.npz')!='3662094d6ec424824c4c3b248d5b674ce14536871686245729da6e333d13d48e':raise ValueError('Graph hash changed')
    np.testing.assert_array_equal(np.load(folder/'past4096_panel_features.npy'),c['features'])
    source,target=graph.row,graph.col;n=graph.shape[0]
    ss,tt,audit=shuffled_edges(source,target,n)
    np.testing.assert_array_equal(np.bincount(ss,minlength=n),np.bincount(source,minlength=n))
    np.testing.assert_array_equal(np.bincount(tt,minlength=n),np.bincount(target,minlength=n))
    np.savez_compressed(run/'kinetic_graphs.npz',source=source,target=target,shuffled_source=ss,shuffled_target=tt)
    mean=np.maximum(c['center']/c['scale']+c['pca'].mean_,0.)
    models={}
    for name,kind,s,t in [('kinetic_none400','none',np.array([],dtype=int),np.array([],dtype=int)),('kinetic_shuffle400','shuffle',ss,tt),('kinetic_real400','real',source,target)]:
        models[name]=KineticFlow(initial,mean,s,t,kind)
    emit('kinetic_graphs_prepared',**audit,edges=len(source),gene_count=n)
    return models,audit
