"""Synthetic resource and equation checks, never a biological score."""
import sys,copy,json,time,threading,traceback,hashlib
from pathlib import Path
import numpy as np
import torch
from scnode_resource_preflight import peak_memory
from threadpoolctl import threadpool_limits
from tigon_conditional_core import ConditionalUOT,integrate,log_density,objective,conditional_weights
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'t1_run'))
from run_t1 import digest
from iterate import append_event,now
HERE=Path(__file__).resolve().parent
RUN=HERE/'private/tigon_numerical_preflight_retry_01';PUBLIC=HERE/'TIGON_NUMERICAL_PREFLIGHT_RETRY_RESULTS.json'

class Analytic:
    def __init__(self,growth):self.growth=growth
    def __call__(self,t,state):
        z,w,p,e=state;v=torch.ones_like(z)*.2;g=torch.ones_like(w)*self.growth
        return v,g,g,(v.square().sum(1,keepdim=True)+g.square())*w.exp()


def main():
    if RUN.exists() or PUBLIC.exists():raise ValueError('No duplicate preflight')
    RUN.mkdir(parents=True);emit=lambda event,**kw:append_event(RUN/'events.jsonl',event,**kw)
    report={'status':'running','raw_metrics':None,'skills':None,'reward_delta':0,'new_scoring_batch':False,'passing_candidates':[]}
    peak=[peak_memory()];stop=threading.Event()
    def monitor():
        while not stop.wait(.05):peak[0]=max(peak[0],peak_memory())
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start();started=time.perf_counter()
    try:
        manifest=json.loads((HERE/'TIGON_AUTHOR_SOURCE_MANIFEST_20261002.json').read_text())
        for record in manifest['files']:
            if digest(HERE/manifest['local_source_directory']/record['path'])!=record['sha256']:raise ValueError('Author source changed')
        plan={'spec_sha256':digest(HERE/'NEXT_TIGON_NUMERICAL_PREFLIGHT.json'),'source_sha256':{f:digest(HERE/f) for f in ['tigon_numerical_preflight_retry.py','tigon_conditional_core.py','scnode_resource_preflight.py']},'author_manifest_sha256':digest(HERE/'TIGON_AUTHOR_SOURCE_MANIFEST_20261002.json'),'seed':20261002,'endpoint_masses':[1.,1.,1.]}
        (RUN/'plan.json').write_text(json.dumps(plan,indent=2));emit('synthetic_plan_frozen',sha256=digest(RUN/'plan.json'))
        z=torch.ones(4,8,dtype=torch.float64)
        for rate in [0.,.3]:
            final,w,p,e=integrate(Analytic(rate),z,0.,.5)
            torch.testing.assert_close(final,z+.1)
            torch.testing.assert_close(w,torch.full_like(w,.5*rate))
            torch.testing.assert_close(p,w)
            expected=(8*.2**2+rate**2)*((np.exp(.5*rate)-1)/rate if rate else .5)
            torch.testing.assert_close(e,torch.full_like(e,expected),atol=1e-5,rtol=1e-5)
            weights,ess=conditional_weights(w);torch.testing.assert_close(weights,torch.ones_like(weights)/len(weights))
        class Affine(ConditionalUOT):
            def fields(self,t,z):return .2*z,z.new_zeros((len(z),1))
        affine=Affine()
        final,w,logp,e=integrate(affine,z,0.,.5)
        torch.testing.assert_close(final,z*np.exp(.1),rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(logp,torch.full_like(logp,-.8),rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(w,torch.zeros_like(w))
        values=torch.randn(5,8,dtype=torch.float64,requires_grad=True);centers=torch.randn(7,8,dtype=torch.float64)
        direct=torch.stack([torch.distributions.MultivariateNormal(c,torch.eye(8,dtype=torch.float64)*.1).log_prob(values) for c in centers],1).logsumexp(1)-np.log(len(centers))
        chunked=log_density(values,centers,chunk=3)
        torch.testing.assert_close(chunked,direct,rtol=1e-10,atol=1e-10)
        a=torch.autograd.grad(chunked.sum(),values,retain_graph=True)[0];b=torch.autograd.grad(direct.sum(),values)[0];torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-9)
        torch.manual_seed(20261002);initial_model=ConditionalUOT();initial=copy.deepcopy(initial_model.state_dict())
        generator=torch.Generator().manual_seed(20261002)
        base=torch.randn(64,8,generator=generator)*.3
        groups=[base,base+.03,base+.06];times=[0.,.25,.5]
        batches=[];stream=hashlib.sha256()
        for step in range(10):
            query=[g[torch.randperm(len(g),generator=generator)[:32]]+torch.randn(32,8,generator=generator)*np.sqrt(.02) for g in groups[1:]]
            start=groups[0][torch.randperm(len(base),generator=generator)[:32]]+torch.randn(32,8,generator=generator)*np.sqrt(.02)
            for v in query+[start]:stream.update(v.numpy().tobytes())
            batches.append((query,start))
        report['arms']={}
        for name,growth in [('growth_disabled',False),('conditional_growth',True)]:
            model=ConditionalUOT(growth=growth);model.load_state_dict(initial)
            optimizer=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=.01);history=[]
            for step,(queries,start) in enumerate(batches):
                loss=objective(model,groups,times,queries,start)
                if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
                optimizer.zero_grad();loss.backward()
                if not all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters()):raise ValueError('Nonfinite gradient')
                velocity_grad=sum(float(v.grad.square().sum()) for v in model.velocity.parameters() if v.grad is not None)
                growth_grad=sum(float(v.grad.square().sum()) for v in model.growth.parameters() if v.grad is not None)
                if velocity_grad<=0 or (growth and growth_grad<=0):raise ValueError('Missing required gradient')
                optimizer.step();history.append(float(loss.detach()))
            final,w,_,_=integrate(model,base,0.,.5);weights,ess=conditional_weights(w)
            torch.testing.assert_close(weights.sum(),weights.new_tensor(1.))
            if not torch.isfinite(final).all() or not 1<=float(ess.detach())<=len(base)+1e-4:raise ValueError('Invalid forecast/ESS')
            if not growth:torch.testing.assert_close(weights,torch.full_like(weights,1/len(base)))
            report['arms'][name]={'loss_history':history,'draw_sha256':stream.hexdigest(),'velocity_gradient_squared_norm':velocity_grad,'growth_gradient_squared_norm':growth_grad,'conditional_weight_ess':float(ess.detach()),'endpoint_mass_policy':'Eachinputmass1; normalizedforecastweights'}
            emit('synthetic_arm_completed',candidate=name,**report['arms'][name])
        report.update(status='completed',checks_passed=True,completed_utc=now(),plan_sha256=digest(RUN/'plan.json'),scope='Syntheticnumerical/capacity checks only. No source expression/author data/weights, no benchmark or biologicalvalidation, no absolute growthclaim.')
    except Exception as exc:
        (RUN/'traceback.txt').write_text(traceback.format_exc());report.update(status='failed',error=type(exc).__name__+': '+str(exc))
    finally:stop.set();watcher.join();peak[0]=max(peak[0],peak_memory())
    report.update(elapsed_seconds=time.perf_counter()-started,observed_peak_process_working_set_bytes=peak[0],rss_sampling_interval_seconds=.05)
    if peak[0]>=16*1024**3:report.update(status='failed',error='16GiBresourcegatefailed')
    (RUN/'report.json').write_text(json.dumps(report,indent=2));report['report_sha256']=digest(RUN/'report.json');PUBLIC.write_text(json.dumps(report,indent=2)+'\n');emit('batch_finished',status=report['status']);print(json.dumps({'status':report['status'],'report':str(PUBLIC)}))

if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
