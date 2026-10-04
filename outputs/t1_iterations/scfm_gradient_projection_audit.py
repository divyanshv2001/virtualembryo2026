"""Bounded frozen past-gradient support for a direction-preserving scFM extension."""
import json
import traceback
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from anchored_soft_ot import combine_gradients, project_auxiliary
from cnf_manifold_flow import DensityFlowNet, inverse_density, rk4_position, manifold_penalty
from soft_ot_flow_matching import soft_coupling
from scnode_past_fold_training import PastOnlyMatrix
from scnode_resource_preflight import peak_memory
from run_t1 import digest
from iterate import now, append_event

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN=HERE/'private/scfm_gradient_projection_audit_01'
PUBLIC=HERE/'SCFM_GRADIENT_PROJECTION_AUDIT_RESULTS.json'


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    if RUN.exists() or PUBLIC.exists():raise ValueError('Never duplicate audit')
    RUN.mkdir(parents=True)
    emit=lambda kind,**kw:append_event(RUN/'events.jsonl',kind,**kw)
    report={'status':'running','panels':[],'new_scoring_batch':False,'reward':0,'passing_candidates':[]}
    try:
        spec=json.loads((HERE/'RESEARCH_HARNESS_MANIFEST.json').read_text())['experiments']['scfm_gradient_projection_audit']['protocol']
        archive=HERE/'private/cnf_feature_challenge_01'
        prepared=HERE/'private/associated_prepared_01'
        historical=json.loads((HERE/'private/cnf_covariance_alignment_01/plan.json').read_text())
        if digest(archive/'features4096.pt')!=historical['flow_sha256'] or digest(archive/'encoder4096.npz')!=historical['encoder_sha256']:
            raise ValueError('Frozen past initialization changed')
        plan={'created_utc':now(),'protocol':spec,'source_sha256':{f:digest(HERE/f) for f in ['scfm_gradient_projection_audit.py','anchored_soft_ot.py','soft_ot_flow_matching.py','cnf_manifold_flow.py','cnf_density_flow.py']},'checkpoint_sha256':historical['flow_sha256'],'encoder_sha256':historical['encoder_sha256']}
        save(RUN/'plan.json',plan);report['plan_sha256']=digest(RUN/'plan.json');emit('plan_frozen',sha256=report['plan_sha256'])
        prep=json.loads((prepared/'report.json').read_text())
        for name,key in [('expression.npy','expression_sha256'),('genes.csv','genes_sha256'),('selected_metadata.csv','metadata_sha256')]:
            if digest(prepared/name)!=prep[key]:raise ValueError('Prepared input changed')
        cutoff=spec['cutoff'];stages=pd.read_csv(prepared/'selected_metadata.csv').numeric_stage.to_numpy(float)
        x=PastOnlyMatrix(np.load(prepared/'expression.npy',mmap_mode='r'),stages,cutoff)
        symbols=pd.read_csv(prepared/'genes.csv').symbol.fillna('').tolist();counts=Counter(symbols)
        lookup={g:i for i,g in enumerate(symbols) if g and counts[g]==1}
        panel=(ROOT/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
        with np.load(archive/'encoder4096.npz') as e:
            features,center,scale=e['features'],e['center'],e['scale']
            net=DensityFlowNet(e['basis'],e['pca_center'],cutoff,7.25)
        net.load_state_dict(torch.load(archive/'features4096.pt',weights_only=False,map_location='cpu')['net']);net.eval()
        columns=np.array([lookup[panel[i]] for i in features]);rows=np.flatnonzero(stages<=cutoff)
        z=np.empty((len(rows),len(net.basis)),dtype=np.float32)
        with torch.no_grad():
            for start in range(0,len(rows),256):
                block=np.asarray(x[np.ix_(rows[start:start+256],columns)])
                z[start:start+256]=net.encode(torch.tensor((block-center)/scale,dtype=torch.float32))[0].numpy()
        if not np.isfinite(z).all():raise ValueError('Nonfinite past latents')
        absolute=np.unique(stages[rows]);times=absolute-net.origin
        groups=[torch.tensor(z[stages[rows]==t]) for t in absolute];reference=torch.tensor(z)
        parameters=list(net.parameters());rng=np.random.default_rng(spec['seed']);torch.manual_seed(spec['seed']);records=[]
        for probe in range(spec['probes']):
            size=spec['batch_size'];group=int(torch.randint(len(groups),(1,)))
            source=groups[group][torch.randint(len(groups[group]),(size,))].clone()
            noise=torch.randint(0,2,source.shape).float()*2-1
            nll,energy=inverse_density(net.velocity,source,float(times[group]),noise=noise,log_base=getattr(net,'log_base',None))
            density=torch.tensor(0.)
            if group>0:density=manifold_penalty(rk4_position(net.velocity,source,float(times[group]),float(times[group])-.125),reference)
            loss=nll.mean()+.1*energy.mean()+10.*density
            base=[torch.zeros_like(p) if g is None else g for p,g in zip(parameters,torch.autograd.grad(loss,parameters,allow_unused=True))]
            pair=int(rng.integers(len(times)-1));left=groups[pair][rng.choice(len(groups[pair]),size,replace=True)].numpy();right=groups[pair+1][rng.choice(len(groups[pair+1]),size,replace=True)].numpy()
            coupling,audit=soft_coupling(left,right);sample=rng.choice(coupling.size,size,replace=True,p=coupling.ravel())
            start=torch.from_numpy(left[sample//size]);finish=torch.from_numpy(right[sample%size]);fraction=torch.tensor(rng.random((size,1)),dtype=torch.float32)
            elapsed=float(times[pair+1]-times[pair]);point=start+fraction*(finish-start);clock=float(times[pair])+fraction*elapsed
            auxiliary_loss=((net.field(torch.cat([point,clock],dim=1))-(finish-start)/elapsed)**2).mean()
            auxiliary=torch.autograd.grad(auxiliary_loss,parameters);projected=project_auxiliary(base,auxiliary)
            gradients,factor,base_norm,projected_norm=combine_gradients(base,projected)
            norm=float(torch.sqrt(sum((g*g).sum() for g in auxiliary)));dot=float(sum((b*a).sum() for b,a in zip(base,auxiliary)));newdot=float(sum((b*a).sum() for b,a in zip(base,projected)))
            tolerance=1e-6*max(base_norm*projected_norm,1.)
            if not all(torch.isfinite(g).all() for g in gradients) or not torch.isfinite(loss) or not torch.isfinite(auxiliary_loss) or newdot < -tolerance:
                raise ValueError('Nonfinite or opposing projected gradient')
            records.append({'probe':probe,'past_stage':float(absolute[group]),'cosine_before':dot/max(base_norm*norm,1e-30),'cosine_after':newdot/max(base_norm*projected_norm,1e-30),'conflict':dot<0,'retained_auxiliary_norm_fraction':projected_norm/max(norm,1e-30),'effective_ot_weight':factor,**audit})
            if (probe+1)%8==0:emit('gradient_probe',completed=probe+1)
        conflicts=sum(r['conflict'] for r in records);retained=float(np.mean([r['retained_auxiliary_norm_fraction'] for r in records]))
        supported=conflicts>=spec['minimum_conflicting_probes'] and retained>=spec['minimum_mean_retained_auxiliary_norm_fraction']
        save(RUN/'probes.json',records)
        report.update(status='completed',completed_utc=now(),scope=spec['scope'],audit_context={'probes':len(records),'conflicting_probes':conflicts,'mean_retained_auxiliary_norm_fraction':retained,'all_projected_gradients_nonopposing_within_tolerance':True,'past_only_max_stage':float(stages[rows].max()),'optimizer_steps':0,'future_expression_reads':0,'score_metrics_available':False,'supports_fixed_training_trial':supported,'limits':'First-order raw-gradient compatibility only; Adam/preconditioning and finite updates may still regress likelihood. No score improvement demonstrated.'})
    except Exception as exc:
        (RUN/'traceback.txt').write_text(traceback.format_exc());report.update(status='failed',error=type(exc).__name__+': '+str(exc),failed_utc=now())
    report['resource_peak_process_working_set_bytes']=peak_memory()
    if peak_memory()>=16*1024**3:report.update(status='failed',error='16GiB resource gate failed')
    save(RUN/'report.json',report);report['report_sha256']=digest(RUN/'report.json');save(PUBLIC,report);emit('batch_finished',status=report['status'])
    print(json.dumps({'status':report['status'],'audit_context':report.get('audit_context'),'error':report.get('error')}))


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
