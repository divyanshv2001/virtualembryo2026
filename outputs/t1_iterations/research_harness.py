"""Central deterministic experiment runner; logs on disk, compact receipts to Codex."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
METRICS=('de_score','de_direction','mmd_u','variogram')

def critics_cached(review,report):
    return (review.get('status')=='completed'
            and review.get('report_sha256')==report.get('report_sha256')
            and review.get('specialist_role_version')==2)

def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(path.read_text()) if path.exists() else {}
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
def resolve(relative):
    path=(HERE/relative).resolve()
    if not path.is_relative_to(HERE):raise ValueError('Manifest path outside project evidence root')
    return path

def processes():
    if os.name!='nt':raise RuntimeError('Live process adapter currently supports Windows only')
    command="Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
    result=subprocess.run(['powershell','-NoProfile','-NonInteractive','-Command',command],capture_output=True,text=True,check=True)
    records=json.loads(result.stdout) if result.stdout.strip() else []
    if isinstance(records,dict):records=[records]
    # Includes venv shim/worker. The harness itself is not experimental work.
    return [r for r in records if r.get('CommandLine') and str(HERE).lower() in r['CommandLine'].lower() and 'research_harness.py' not in r['CommandLine'].lower()]

def status(name,entry,live):
    report=resolve(entry['report']);run=resolve(entry['run']);receipt=read(HERE/'private/harness_runs'/name/'receipt.json')
    matching=[r for r in live if str(resolve(entry['script'])).lower().replace('/',chr(92)) in r['CommandLine'].lower().replace('/',chr(92)) and (all(str(arg).lower() in r['CommandLine'].lower() for arg in entry['args']) if entry.get('args') else '--experiment' not in r['CommandLine'].lower())]
    if matching:return {'experiment':name,'status':'running','pids':[r['ProcessId'] for r in matching]}
    if report.exists():return {'experiment':name,'status':'completed','report':entry['report'],'report_sha256':digest(report)}
    if run.exists() or receipt:return {'experiment':name,'status':'interrupted_or_failed','evidence':entry['run'],'reason':'Process absent and final report missing; inspect saved stderr/events before recovery.'}
    return {'experiment':name,'status':'not_started'}

def _run(name,entry,live):
    existing=status(name,entry,live)
    if existing['status']!='not_started':raise ValueError('Do not duplicate or overwrite existing run: '+existing['status'])
    if live:raise ValueError('Other project Python work active; inspect status first')
    if HERE.drive.upper()!='D:':raise ValueError('Experiment/cache must stay on D:')
    script=resolve(entry['script'])
    if not script.is_file():raise ValueError('Registered script missing')
    folder=HERE/'private/harness_runs'/name;folder.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (folder/'stdout.log').open('wb') as stdout,(folder/'stderr.log').open('wb') as stderr:
        worker=subprocess.Popen([sys.executable,'-u',str(script)]+entry.get('args',[]),cwd=HERE.parents[1],stdout=stdout,stderr=stderr,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
    receipt={'experiment':name,'started_utc':now(),'pid':worker.pid,'script_sha256':digest(script),'registered_run':entry['run'],'registered_report':entry['report'],'logs':str(folder.relative_to(HERE)),'status':'started'}
    save(folder/'receipt.json',receipt)
    return {'experiment':name,'status':'started','pid':worker.pid,'logs':receipt['logs']}

def run(name,entry,live):
    directory=HERE/'private/harness_runs';directory.mkdir(parents=True,exist_ok=True)
    lock=directory/'launch.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(fd,str(os.getpid()).encode())
        return _run(name,entry,processes())
    finally:
        os.close(fd);lock.unlink()

def collect(name,entry,live):
    state=status(name,entry,live)
    if state['status']!='completed':return state
    folder=HERE/'private/harness_runs'/name;collected=read(folder/'collected.json')
    checkpoint=read(HERE/'LOCAL_OPTIMIZATION_STATE.json')
    previous_review=checkpoint.get('post_batch_agent_critiques',{}).get(Path(entry['run']).name,{})
    report=read(resolve(entry['report']))
    critics_complete=critics_cached(previous_review,report)
    if collected.get('report_sha256')==state['report_sha256']:return {'experiment':name,'status':'unchanged','new_critic_packets':0,'critic_agents_needed':not critics_complete}
    report=read(resolve(entry['report']));packet_paths=[]
    summary=report.get('summary',[])
    # Emit only headlines/pointers. Four raw metrics/skills remain in original report/ledger and per-metric packets.
    for metric in METRICS:
        path=folder/('critic_'+metric+'.json')
        command=[sys.executable,str(HERE/'build_critic_packet.py'),'--report',str(resolve(entry['report'])),'--metric',metric,'--out',str(path)]
        subprocess.run(command,cwd=HERE.parents[1],capture_output=True,text=True,check=True)
        packet_paths.append(str(path.relative_to(HERE)))
    receipt={**state,'collected_utc':now(),'scores':[{'candidate':r.get('candidate'),'scores':r.get('scores'),'all_calibrations_valid':r.get('all_calibrations_valid')} for r in summary],
        'passing_candidates':report.get('passing_candidates'),'critic_packets':packet_paths,'critics_complete':critics_complete,'critic_agents_needed':not critics_complete,'scope':'Collection does not execute agents, call Jev, certify promotion or alter rewards.'}
    save(folder/'collected.json',receipt)
    return receipt

def inline_critics(name,entry):
    report=read(resolve(entry['report']))
    checkpoint=read(HERE/'LOCAL_OPTIMIZATION_STATE.json')
    review=checkpoint.get('post_batch_agent_critiques',{}).get(Path(entry['run']).name,{})
    if critics_cached(review,report):return {'experiment':name,'status':'cached','critic_agents_needed':False}
    report_hash=digest(resolve(entry['report']))
    packets={}
    for metric in METRICS:
        packet=read(HERE/'private/harness_runs'/name/('critic_'+metric+'.json'))
        if packet.get('report_sha256')!=report_hash:raise ValueError('Collect current report before requesting inline critics')
        packets[metric]=packet
    return {'experiment':name,'report_sha256':report_hash,'critic_agents_needed':True,
            'delivery':'Spawn four fresh metric-only reviewers with their packet inline; no tools, files, peers or writes; <=60words. Coordinator validates facts.',
            'packets':packets}

def advise(name,entry,decision,options,dry_run=False):
    import re
    if not decision or not re.fullmatch(r'[a-z0-9_-]{1,64}',decision):raise ValueError('Use a short stable decision id')
    choices=options if isinstance(options,dict) else json.loads(options or '{}')
    if not isinstance(choices,dict) or not 2<=len(choices)<=6 or not all(isinstance(v,str) and len(v)<=300 for v in choices.values()):raise ValueError('Provide 2-6 concise options with criteria')
    report=read(resolve(entry['report']))
    if report.get('status') not in ('completed','failed'):raise ValueError('Advisory requires final preserved evidence')
    checkpoint=read(HERE/'LOCAL_OPTIMIZATION_STATE.json')
    review=checkpoint.get('post_batch_agent_critiques',{}).get(Path(entry['run']).name,{})
    packet={'model':'jev-1.13.0','state':{'decision':decision,'report_sha256':digest(resolve(entry['report'])),
        'status':report.get('status'),'error':report.get('error'),'passing':report.get('passing_candidates'),
        'summary':[{'candidate':r.get('candidate'),'mean':r.get('mean_score'),'skills':r.get('mean_skills'),'valid':r.get('all_calibrations_valid')} for r in report.get('summary',[])],
        'critic_problems':{m:v.split('Proposed solution:')[0] for m,v in review.get('reviews',{}).items()} if isinstance(review.get('reviews'),dict) else {},
        'constraints':'Past-only fit; unchanged full-panel scorer/calibration/gates. Previously exposed development is not independent validation. No target tuning, repeat, absolute-growth claim or official upload; remedies unvalidated.'},
        'questions':{'next':{'type':'choice','criteria':choices}}}
    size=len(json.dumps(packet,separators=(',',':'),sort_keys=True).encode())
    if size>3000:raise ValueError('Advisory exceeds3000bytes; narrow options rather than transmit more context')
    if dry_run:return {'experiment':name,'decision':decision,'payload_bytes':size,'requests':0,'mode':'dry_run'}
    folder=HERE/'private/harness_runs'/name/'decisions';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(decision+'_packet.json');record=folder/(decision+'_route.json')
    if path.exists() and read(path)!=packet:raise ValueError('Decision id already has different evidence; preserve original')
    if not path.exists():save(path,packet)
    result=subprocess.run([sys.executable,str(HERE/'jev_compact_route.py'),'--packet',str(path),'--record',str(record)],cwd=HERE.parents[1],capture_output=True,text=True,check=True)
    return json.loads(result.stdout)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['status','run','collect','queue','resume','wait','critics','advise']);parser.add_argument('--experiment');parser.add_argument('--wait-seconds',type=int,default=20);parser.add_argument('--decision');parser.add_argument('--options');parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    if args.command=='resume':
        checkpoint=read(HERE/'LOCAL_OPTIMIZATION_STATE.json')
        manifest=read(HERE/'RESEARCH_HARNESS_MANIFEST.json')['experiments']
        latest=read(HERE/'METRIC_RESEARCH_QUEUE.json').get('latest_paper_decision',{})
        live=processes();efficiency=checkpoint.get('efficiency_policy',{})
        policy_hash=digest(HERE/'RESEARCH_CONTINUATION_POLICY.md')
        from metric_critique_reward import health_snapshot
        ledger=HERE/'METRIC_CRITIQUE_REWARD_LEDGER.jsonl'
        health=health_snapshot([json.loads(line) for line in ledger.read_text().splitlines() if line.strip()])
        names=[args.experiment] if args.experiment else checkpoint.get('active_jobs',[])
        if not names and latest.get('experiment'):names=[latest['experiment']]
        result={'policy_changed':policy_hash!=efficiency.get('policy_sha256'),
                'live_workers':len(live),
                'agent_health':{k:health[k] for k in ('score','redline','status')},
                'jobs':[status(n,manifest[n],live) if n in manifest else {'experiment':n,'status':'not_registered'} for n in names],
                'next_experiment':latest.get('next_experiment'),
                'next_action':latest.get('next') or checkpoint.get('next_experiment'),
                'efficiency':{k:efficiency.get(k) for k in ('routine_cycle_soft_target_percentage_points','rules')},
                'read_if_changed':'RESEARCH_CONTINUATION_POLICY.md',
                'scope':'Read-only compact resume; no experiments, critics, Jev or files written.'}
        print(json.dumps(result,separators=(',',':')));return
    if args.command=='queue':
        queue=read(HERE/'METRIC_RESEARCH_QUEUE.json')
        latest=queue.get('latest_paper_decision',{})
        pending=[p for p in queue.get('paths',[]) if p.get('status') in ('predeclared','registered','not_started','implementation_pending')]
        result={'latest':{k:latest.get(k) for k in ('status','implementation_status','next_experiment','next')},
                'pending_count':len(pending),'pending':[{'id':p.get('id'),'status':p.get('status'),'next_action':str(p.get('next_action',''))[:200]} for p in pending[:5]],
                'scope':'Completed-family histories and nested results omitted; open-family flags do not authorize repeats.'}
        print(json.dumps(result,separators=(',',':')));return
    manifest=read(HERE/'RESEARCH_HARNESS_MANIFEST.json')['experiments'];live=processes()
    if args.command=='status':
        if args.experiment and args.experiment not in manifest:raise ValueError('Unknown registered experiment')
        selected={args.experiment:manifest[args.experiment]} if args.experiment else manifest
        result={'live_project_python_processes':len(live),'experiments':[status(name,entry,live) for name,entry in selected.items()]}
    else:
        if args.experiment not in manifest:raise ValueError('Register experiment in manifest before execution')
        entry=manifest[args.experiment]
        if args.command=='critics':result=inline_critics(args.experiment,entry)
        elif args.command=='advise':result=advise(args.experiment,entry,args.decision,args.options,args.dry_run)
        elif args.command=='wait':
            if not 0<=args.wait_seconds<=45:raise ValueError('Wait must be between0and45seconds')
            deadline=time.monotonic()+args.wait_seconds
            while True:
                result=status(args.experiment,entry,live)
                if result['status']!='running' or time.monotonic()>=deadline:break
                time.sleep(min(5,max(0,deadline-time.monotonic())));live=processes()
        else:result=(run if args.command=='run' else collect)(args.experiment,entry,live)
    print(json.dumps(result,separators=(',',':')))

if __name__=='__main__':
    try:main()
    except ValueError as exc:
        print(json.dumps({'status':'blocked','reason':str(exc)},separators=(',',':')));sys.exit(2)
