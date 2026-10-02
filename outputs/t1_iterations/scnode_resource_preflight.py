"""Synthetic capacity/numerical preflight only; no biological score or reward."""
import copy,ctypes,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from scnode_joint_model import JointModel,joint_loss
from iterate import append_event,now
from run_t1 import digest

HERE=Path(__file__).resolve().parent
RUN=HERE/'private/scnode_resource_preflight_01'
PUBLIC=HERE/'SCNODE_RESOURCE_PREFLIGHT_RESULTS.json'


class Memory(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]


def peak_memory():
    m=Memory();m.cb=ctypes.sizeof(m)
    kernel=ctypes.windll.kernel32;kernel.GetCurrentProcess.restype=ctypes.c_void_p
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(kernel.GetCurrentProcess()),ctypes.byref(m),m.cb): raise OSError('Memory measurement failed')
    return m.PeakWorkingSetSize


def main():
    if RUN.exists() or PUBLIC.exists(): raise ValueError('No duplicate preflight')
    RUN.mkdir(parents=True);start=time.monotonic()
    emit=lambda event,**kw:append_event(RUN/'events.jsonl',event,**kw)
    plan={'seed':20261002,'synthetic_genes':32285,'cells_per_stage':64,'past_stages':[7.25,7.5,7.75,8.,8.25,8.5],'latent':32,'width':128,'batch':32,'pretrain_steps':5,'joint_steps_per_arm':3,'betas':[0.,.1],'lr':.001,'adam_betas':[.95,.99],'no_biological_data':True,'no_benchmark_scores':True,'source_sha256':{f:digest(HERE/f) for f in ['scnode_joint_model.py','scnode_resource_preflight.py']}}
    (RUN/'plan.json').write_text(json.dumps(plan,indent=2))
    report={'status':'running','plan_sha256':digest(RUN/'plan.json'),'raw_metrics':None,'skills':None,'passing_candidates':[],'new_scoring_batch':False,'reward_delta':0}
    try:
        torch.manual_seed(plan['seed']);rng=np.random.default_rng(plan['seed'])
        data=[torch.tensor(rng.lognormal(-2.,.8,(64,32285)).astype(np.float32)) for _ in plan['past_stages']]
        model=JointModel(32285);optimizer=torch.optim.Adam(list(model.encoder.parameters())+list(model.mu.parameters())+list(model.std.parameters())+list(model.decoder.parameters()),lr=.001,betas=(.95,.99))
        for step in range(5):
            batch=data[step%len(data)][:32];z=model.sample(batch,torch.randn(32,32));loss=((model.decoder(z)-batch)**2).mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
        initial=copy.deepcopy(model.state_dict());history=[]
        times=torch.tensor(plan['past_stages'])-7.25
        for beta in plan['betas']:
            model.load_state_dict(initial);torch.manual_seed(plan['seed']);rng=np.random.default_rng(plan['seed'])
            optimizer=torch.optim.Adam(model.parameters(),lr=.001,betas=(.95,.99))
            for step in range(3):
                batches=[x[rng.choice(len(x),32,replace=False)] for x in data]
                noises=[torch.randn(32,32) for _ in range(len(data)+1)]
                loss,expression,dynamic=joint_loss(model,batches,times,noises,beta)
                if not torch.isfinite(loss): raise ValueError('Nonfinite joint loss')
                optimizer.zero_grad();loss.backward()
                norms={name:float(torch.sqrt(sum((p.grad*p.grad).sum() for p in module.parameters() if p.grad is not None))) for name,module in [('encoder',model.encoder),('decoder',model.decoder),('drift',model.drift)]}
                if not all(np.isfinite(v) and v>0 for v in norms.values()): raise ValueError('Missing/nonfinite component gradient')
                optimizer.step();peak=peak_memory()
                if peak>4*1024**3: raise ValueError('Preflight process exceeds4GiB budget')
                record={'beta':beta,'step':step+1,'loss':float(loss.detach()),'expression_loss':float(expression.detach()),'dynamic_loss':float(dynamic.detach()),'gradient_norms':norms,'peak_working_set_bytes':peak}
                history.append(record);emit('synthetic_joint_step',**record)
        report.update(status='completed',completed_utc=now(),history=history,peak_working_set_bytes=peak_memory(),elapsed_seconds=time.monotonic()-start,capacity_preflight_passed=True,scientific_training_ready=False,scope='Synthetic forward/backward and memory checks only. No biological accuracy/decoder transfer validation; detailed past-only fullgene adaptation remains required.')
    except Exception as exc:
        (RUN/'traceback.txt').write_text(traceback.format_exc());report.update(status='failed',error=type(exc).__name__+': '+str(exc))
    (RUN/'report.json').write_text(json.dumps(report,indent=2));report['report_sha256']=digest(RUN/'report.json');PUBLIC.write_text(json.dumps(report,indent=2));emit('batch_finished',status=report['status']);print(json.dumps({'status':report['status']}))


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2): main()
