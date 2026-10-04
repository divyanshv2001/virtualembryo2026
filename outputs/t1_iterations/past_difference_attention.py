"""Own past-centroid difference attention adaptation, not scDiformer reproduction."""
import copy
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from cnf_manifold_flow import DensityFlowNet,rk4_position
from graph_kinetic_residual import KineticFlow

HERE=Path(__file__).resolve().parent

class PastAttentionFlow(torch.nn.Module):
    def __init__(self,base,times,centroids,kind):
        super().__init__()
        if kind not in ('level','difference'):raise ValueError('Unknown attention input')
        if max(times)>base.cutoff:raise ValueError('Future memory forbidden')
        self.base=copy.deepcopy(base).eval();self.cutoff=base.cutoff;self.origin=base.origin;self.kind=kind
        for p in self.base.parameters():p.requires_grad_(False)
        self.register_buffer('times',torch.tensor(times,dtype=torch.float32))
        self.register_buffer('centroids',torch.tensor(centroids,dtype=torch.float32))
        self.embed=torch.nn.Linear(8,32);self.attention=torch.nn.MultiheadAttention(32,4,batch_first=True,dropout=0.)
        self.output=torch.nn.Linear(32,8)
        torch.nn.init.zeros_(self.output.weight);torch.nn.init.zeros_(self.output.bias)
        self.context_stage=self.cutoff

    def encode(self,x):return self.base.encode(x)

    def memory(self):
        values=self.centroids[self.times<=self.context_stage]
        if not len(values):raise ValueError('Empty causal memory')
        if self.kind=='difference':
            intervals=self.times[self.times<=self.context_stage]
            values=torch.cat([torch.zeros_like(values[:1]),(values[1:]-values[:-1])/(intervals[1:]-intervals[:-1])[:,None]])
        return values

    def velocity(self,t,z):
        tokens=self.embed(self.memory())[None].expand(len(z),-1,-1)
        query=self.embed(z)[:,None]
        hidden,_=self.attention(query,tokens,tokens,need_weights=False)
        residual=self.output(hidden[:,0])
        residual=residual/torch.clamp(torch.linalg.vector_norm(residual,dim=1,keepdim=True),min=1.)
        return self.base.velocity(t,z)+residual

    def trajectory(self,z,times,step=.125):
        self.context_stage=self.cutoff;history=[z]
        for a,b in zip(times[:-1],times[1:]):
            z=rk4_position(self.velocity,z,self.cutoff-self.origin+float(a),self.cutoff-self.origin+float(b),step)
            history.append(z)
        return torch.stack(history)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',default='difference_attention_preflight');args=parser.parse_args()
    entry=json.loads((HERE/'RESEARCH_HARNESS_MANIFEST.json').read_text())['experiments'][args.experiment]
    run=HERE/entry['run'];public=HERE/entry['report']
    if run.exists() or public.exists():raise ValueError('Never repeat preflight')
    run.mkdir();torch.set_num_threads(2)
    assert torch.cuda.is_available(),'Declared CUDA unavailable'
    torch.cuda.set_per_process_memory_fraction(.75,0)
    previous=HERE/'private/graph_kinetics_past_fold_repair_01'
    e=dict(np.load(previous/'fresh_encoder.npz'))
    data=HERE/'private/associated_prepared_01'
    stages=pd.read_csv(data/'selected_metadata.csv').numeric_stage.to_numpy(float)
    symbols=pd.read_csv(data/'genes.csv').symbol.fillna('').tolist()
    panel=(HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
    lookup={g:i for i,g in enumerate(symbols)};atlas=np.array([lookup[panel[i]] for i in e['features']])
    past=np.flatnonzero(stages<=8.25);times=np.unique(stages[past]);assert times.tolist()==[7.5,7.75,8.,8.25]
    raw=np.load(data/'expression.npy',mmap_mode='r')
    values=np.asarray(raw[np.ix_(past,atlas)],np.float32)
    z=((values-e['center'])/e['scale']-e['pca_center'])@e['basis'].T
    centroids=np.stack([z[stages[past]==t].mean(0) for t in times])
    base=DensityFlowNet(e['basis'],e['pca_center'],8.25,7.25)
    saved=torch.load(previous/'kinetic_none400.pt',weights_only=False,map_location='cpu')
    base=KineticFlow(base,saved['net']['gene_mean'].numpy(),np.array([],int),np.array([],int),'none');base.load_state_dict(saved['net'])
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    plan={'cutoff':8.25,'times':times.tolist(),'kind':'resource_preflight_only','optimizer_steps':2,'batch':64,'device':'cuda:0','checkpoint_sha256':digest(previous/'kinetic_none400.pt'),'encoder_sha256':digest(previous/'fresh_encoder.npz'),'scope':'Past centroid attention; no individual trajectories or author reproduction. No target expression or scientific score.'}
    (run/'plan.json').write_text(json.dumps(plan,indent=2))
    reports=[];start=time.perf_counter()
    for kind in ('level','difference'):
        torch.manual_seed(20261004);model=PastAttentionFlow(base,times,centroids,kind).cuda()
        model.context_stage=8.;assert len(model.memory())==3
        a=torch.tensor(z[stages[past]==8.][:64],device='cuda');b=torch.tensor(z[stages[past]==8.25][:64],device='cuda')
        with torch.no_grad():torch.testing.assert_close(model.velocity(.75,a),model.base.velocity(.75,a),rtol=0,atol=0)
        opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
        losses=[]
        for _ in range(2):
            opt.zero_grad();pred=rk4_position(model.velocity,a,.75,1.,step=.125);loss=((pred-b)**2).mean()
            assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
            opt.step();losses.append(float(loss.detach()))
        path=run/(kind+'.pt');torch.save(model.state_dict(),path)
        clone=copy.deepcopy(model);clone.load_state_dict(torch.load(path,weights_only=True,map_location='cuda'))
        with torch.no_grad():torch.testing.assert_close(model.velocity(.75,a),clone.velocity(.75,a),rtol=0,atol=0)
        reports.append({'kind':kind,'losses':losses,'learned_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'checkpoint_replay_exact':True})
        del model,clone,opt
    torch.cuda.synchronize();assert reports[0]['learned_parameters']==reports[1]['learned_parameters']
    report={'status':'completed','new_scoring_batch':False,'plan':plan,'arms':reports,'seconds':time.perf_counter()-start,'peak_allocated_mib':torch.cuda.max_memory_allocated()//2**20,'scope':'Resource/optimizer smoke only; no predictive score, reward or readiness claim.'}
    public.write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
