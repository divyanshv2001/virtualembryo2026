"""Own evolving-population attention audit; not an scIMF reproduction."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from cnf_manifold_flow import DensityFlowNet,rk4_position
from graph_kinetic_residual import KineticFlow
from past_difference_attention import PastAttentionFlow
from geomloss import SamplesLoss

HERE=Path(__file__).resolve().parent

class CollectiveContextFlow(PastAttentionFlow):
    def __init__(self,base,kind):
        if kind not in ('frozen','evolving'):raise ValueError('Unknown population context')
        super().__init__(base,[base.cutoff],np.zeros((1,8),np.float32),'level')
        self.kind=kind;self.query_chunk=128;self.context_budget=256
        self.initial_context=None;self.context_indices=None

    def set_context(self,z):
        if len(z)<2:raise ValueError('At least two context cells needed')
        indices=np.random.default_rng(20261004).permutation(len(z))[:min(self.context_budget,len(z))]
        self.context_indices=torch.tensor(indices,device=z.device)
        self.initial_context=z.detach().clone()

    def velocity(self,t,z):
        if self.initial_context is None or len(self.initial_context)!=len(z):raise ValueError('Population context not initialized')
        pool=(z if self.kind=='evolving' else self.initial_context)[self.context_indices]
        tokens=self.embed(pool)[None];parts=[]
        for start in range(0,len(z),self.query_chunk):
            end=min(start+self.query_chunk,len(z));query=self.embed(z[start:end])[None]
            mask=torch.arange(start,end,device=z.device)[:,None]==self.context_indices[None,:]
            h,_=self.attention(query,tokens,tokens,attn_mask=mask,need_weights=False)
            parts.append(self.output(h[0]))
        residual=torch.cat(parts)
        residual=residual/torch.clamp(torch.linalg.vector_norm(residual,dim=1,keepdim=True),min=1.)
        return self.base.velocity(t,z)+residual

    def trajectory(self,z,times,step=.125):
        self.set_context(z)
        return super().trajectory(z,times,step)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',default='collective_context_preflight');parser.add_argument('--train-packet');args=parser.parse_args()
    if args.train_packet:
        train_packet(Path(args.train_packet));return
    entry=json.loads((HERE/'RESEARCH_HARNESS_MANIFEST.json').read_text())['experiments'][args.experiment]
    run=HERE/entry['run'];public=HERE/entry['report']
    if run.exists() or public.exists():raise ValueError('Never duplicate preflight')
    run.mkdir();assert torch.cuda.is_available();torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.75,0)
    prior=HERE/'private/graph_kinetics_past_fold_repair_01';e=dict(np.load(prior/'fresh_encoder.npz'))
    data=HERE/'private/associated_prepared_01';stages=pd.read_csv(data/'selected_metadata.csv').numeric_stage.to_numpy(float)
    symbols=pd.read_csv(data/'genes.csv').symbol.fillna('').tolist()
    panel=(HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines();lookup={g:i for i,g in enumerate(symbols)}
    columns=np.array([lookup[panel[i]] for i in e['features']]);x=np.load(data/'expression.npy',mmap_mode='r')
    def coords(stage):
        rows=np.flatnonzero(stages==stage)[:128];v=np.asarray(x[np.ix_(rows,columns)],np.float32)
        return torch.tensor(((v-e['center'])/e['scale']-e['pca_center'])@e['basis'].T,device='cuda')
    a,b=coords(8.),coords(8.25)
    saved=torch.load(prior/'kinetic_none400.pt',map_location='cpu',weights_only=False)
    base=KineticFlow(DensityFlowNet(e['basis'],e['pca_center'],8.25,7.25),saved['net']['gene_mean'].numpy(),np.array([],int),np.array([],int),'none');base.load_state_dict(saved['net'])
    plan={'source_commit':'5b763a719a6a87a15a290ab7707850aaba05d5b6','cutoff':8.25,'source_stage':8.,'destination_stage':8.25,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'kind':'resource_only','context_cap':256,'query_chunk':128,'seed':20261004,'scope':'Own frozen versus evolving context residual. No author code copied; no target score, spatial/causal claim or readiness.'}
    (run/'plan.json').write_text(json.dumps(plan,indent=2));out=[];start=time.perf_counter()
    for kind in ('frozen','evolving'):
        torch.manual_seed(20261004);model=CollectiveContextFlow(base,kind).cuda();model.set_context(a)
        with torch.no_grad():torch.testing.assert_close(model.velocity(.75,a),model.base.velocity(.75,a),rtol=0,atol=0)
        opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001);losses=[]
        for _ in range(3):
            model.set_context(a);opt.zero_grad();pred=rk4_position(model.velocity,a,.75,1.,step=.125);loss=((pred-b)**2).mean()
            assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
            opt.step();losses.append(float(loss.detach()))
        with torch.no_grad():
            model.set_context(a);model.query_chunk=128;full=model.velocity(.75,a)
            model.query_chunk=32;chunked=model.velocity(.75,a);chunk_error=float((full-chunked).abs().max());torch.testing.assert_close(full,chunked,rtol=1e-5,atol=1e-6)
            model.query_chunk=128;model.context_budget=64;model.set_context(a);subset=model.velocity(.75,a);subset_error=float((full-subset).abs().max())
            assert torch.isfinite(subset).all();model.context_budget=256;model.set_context(a)
        path=run/(kind+'.pt');torch.save(model.state_dict(),path);clone=copy.deepcopy(model);clone.load_state_dict(torch.load(path,map_location='cuda',weights_only=True))
        with torch.no_grad():torch.testing.assert_close(model.velocity(.75,a),clone.velocity(.75,a),rtol=0,atol=0)
        out.append({'kind':kind,'parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'losses':losses,'query_chunk_max_error':chunk_error,'context_subset_max_error':subset_error,'checkpoint_replay_exact':True})
    torch.cuda.synchronize();assert out[0]['parameters']==out[1]['parameters']
    report={'status':'completed','new_scoring_batch':False,'plan':plan,'arms':out,'seconds':time.perf_counter()-start,'gpu_peak_mib':torch.cuda.max_memory_allocated()/2**20,'scope':'Finite gradient/runtime and query-chunk checks; context subset sensitivity measured, not biological validation or predictive score.'}
    public.write_text(json.dumps(report,indent=2));print(json.dumps(report))

def train_packet(path):
    packet=json.loads(path.read_text());run=path.parent
    if not run.resolve().is_relative_to(HERE/'private'):raise ValueError('GPU packet outside D evidence')
    for filename,expected in packet['sha256'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=expected:raise ValueError('GPU input/source hash changed')
    if not torch.cuda.is_available() or torch.__version__!='2.11.0+cu128':raise ValueError('Pinned CUDA runtime required')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.75,0);torch.use_deterministic_algorithms(True)
    c=dict(np.load(packet['context']));stages=c['stages'];times=np.unique(stages)
    if times.tolist()!=[7.5,7.75,8.,8.25]:raise ValueError('Past-stage support changed')
    saved=torch.load(packet['kinetic_checkpoint'],weights_only=False,map_location='cpu')
    if saved['kind']!='none' or saved['steps']!=400 or saved['fit_max_stage']!=8.25:raise ValueError('Frozen base metadata changed')
    base=KineticFlow(DensityFlowNet(c['basis'],c['pca_center'],8.25,7.25),saved['net']['gene_mean'].numpy(),np.array([],int),np.array([],int),'none');base.load_state_dict(saved['net'])
    frozen={k:v.clone() for k,v in base.state_dict().items()}
    groups=[torch.tensor(c['coordinates'][stages==t],dtype=torch.float32,device='cuda') for t in times]
    loss_fn=SamplesLoss('sinkhorn',p=2,blur=.05,scaling=.9,backend='tensorized')
    def emit(value):
        from datetime import datetime,timezone
        value['timestamp_utc']=datetime.now(timezone.utc).isoformat()
        with (run/'events.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    for kind in ('frozen','evolving'):
        checkpoint=run/('collective_'+kind+'400.pt')
        if checkpoint.exists():raise ValueError('Never retrain completed arm')
        torch.manual_seed(20261004);model=CollectiveContextFlow(base,kind).cuda()
        parameters=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.Adam(parameters,lr=.001)
        rng=torch.Generator().manual_seed(20261004);history=[]
        for i in range(400):
            j=int(torch.randint(len(groups)-1,(1,),generator=rng))
            a=groups[j][torch.randint(len(groups[j]),(64,),generator=rng).cuda()]
            b=groups[j+1][torch.randint(len(groups[j+1]),(64,),generator=rng).cuda()]
            model.set_context(a)
            pred=rk4_position(model.velocity,a,float(times[j]-model.origin),float(times[j+1]-model.origin),step=.125)
            loss=loss_fn(pred,b)
            if not torch.isfinite(loss):raise ValueError('Nonfinite collective loss')
            opt.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,5.)
            if not torch.isfinite(norm):raise ValueError('Nonfinite collective gradient')
            opt.step()
            if (i+1)%50==0:
                record={'event':'collective_training_checkpoint','context_kind':kind,'step':i+1,'loss':float(loss.detach()),'source_stage':float(times[j]),'destination_stage':float(times[j+1])}
                history.append(record.copy());emit(record)
        torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()
        if peak>4.5*1024**3:raise ValueError('GPU allocation cap exceeded')
        model.cpu().eval()
        for k,v in model.base.state_dict().items():torch.testing.assert_close(v,frozen[k],rtol=0,atol=0)
        torch.save({'net':model.state_dict(),'kind':kind,'steps':400,'batch_size':64,'fit_max_stage':8.25,'history':history,'frozen_base_exact':True,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'peak_allocated_bytes':peak,'parameters':sum(p.numel() for p in parameters)},checkpoint)
        emit({'event':'collective_arm_completed','context_kind':kind,'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
        del model,opt
    for filename,expected in packet['sha256'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=expected:raise ValueError('GPU source changed during training')
    print(json.dumps({'status':'completed','trained_device':'RTX3060','steps_per_arm':400}))

if __name__=='__main__':main()
