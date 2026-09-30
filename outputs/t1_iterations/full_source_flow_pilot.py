"""Matched sampled/full-source 8D flow pilot on strictly earlier source stages."""
import argparse
import json
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from train_extended_atlas import HERE
from cnf_density_flow import DensityFlowNet,train_density
from offline_backtest import load_core
from run_t1 import digest
from iterate import now,append_event


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    coords=HERE/'private/full_source_latent_coordinates_01'
    old=HERE/'private/cnf_feature_challenge_01'
    sample=HERE/'private/associated_prepared_01'
    out=HERE/'private/full_source_flow_pilot_01'
    if out.exists() and not args.resume:raise ValueError('Preserve prior pilot; use --resume')
    c_report=json.loads((coords/'report.json').read_text())
    if c_report.get('status')!='completed':raise ValueError('Require completed coordinates')
    plan={'created_utc':now(),'source_sha256':{f:digest(HERE/f) for f in ['full_source_flow_pilot.py','cnf_density_flow.py','full_source_latent_coordinates.py']},
          'coordinates_sha256':digest(coords/'coordinates.npz'),
          'coordinate_report_sha256':digest(coords/'report.json'),
          'encoder_sha256':digest(old/'encoder4096.npz'),
          'sample_row_sha256':digest(sample/'source_rows.npy'),
          'cutoff':8.,'targets':[8.25,8.5],
          'step_budget':200,'batch_size':64,'energy_weight':.1,'density_weight':0.,
          'seed':20260928,'donor_seed':20260929,'donor_count':1500,'target_count':1000,
          'fit':'Frozen 8D encoder; exactly matched 200-update tanh CNFs with the existing 64-cell stage-uniform likelihood/energy trainer, once on the sampled past source cells and once on all past source cells <=E8.0. No target rows in fit. This is a low-cost exposure pilot, not full-cell coverage.',
          'evaluation':'Frozen source E8.0 donor rows. Lower 8D unbiased multi-kernel MMD versus held-out source E8.25/E8.5 cells. Persistence and sampled-source field are controls. Source latent metric only, not 32,285-gene challenge score.',
          'promotion_rule':'Consider a full-coverage source fit only if full-source MMD is lower than sampled-source and persistence on both targets. No E9.5 challenge forecast from a failed pilot.',
          'gate':'Even a positive pilot requires a full-panel historical check and the original >72 mean/lower-tail 64-replicate temporal readiness gate before official consideration.',
          'storage':'D-only small checkpoints/reports/indices; no full expression or forecast matrix.',
          'submissions_allowed':0,'jev_requests_allowed':0}
    if args.resume:
        prior=json.loads((out/'plan.json').read_text())
        for key in plan:
            if key!='created_utc' and prior[key]!=plan[key]:raise ValueError('Resume plan mismatch: '+key)
        plan=prior
    else:
        out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2))
    events=out/'events.jsonl';append_event(events,'pilot_resumed' if args.resume else 'plan_frozen',sha256=digest(out/'plan.json'))
    state_path=HERE/'LOCAL_OPTIMIZATION_STATE.json';state=json.loads(state_path.read_text())
    state.setdefault('full_source_flow_pilot_job',{}).update(status='running',run=out.name,
        plan_sha256=digest(out/'plan.json'),step_budget=200,full_cell_coverage=False)
    state['active_jobs']=[out.name];state['active_run_path']=f'private/{out.name}'
    state['local_process_running']=True
    state['next_experiment']='Matched sampled/full past-source 8D CNF pilot running; no duplicate work or E9.5 forecast until historical source targets are scored.'
    state_path.write_text(json.dumps(state,indent=2))
    with np.load(coords/'coordinates.npz') as saved:
        z=saved['z'].astype(np.float32);stages=saved['stages'];source_rows=saved['source_rows']
    if len(z)!=c_report['source_rows']:raise ValueError('Coordinate row count changed')
    overlap=np.isin(source_rows,np.load(sample/'source_rows.npy'))
    encoder=np.load(old/'encoder4096.npz')
    fits={'sampled':(z[overlap & (stages<=8.)],stages[overlap & (stages<=8.)]),
          'full':(z[stages<=8.],stages[stages<=8.])}
    histories={};nets={}
    for name,(training,training_stages) in fits.items():
        torch.manual_seed(plan['seed'])
        net=DensityFlowNet(encoder['basis'],encoder['pca_center'],8.,7.25)
        history=train_density(net,training,training_stages,.1,out/f'{name}.pt',
            lambda kind,**kw:append_event(events,kind,source=name,**kw),
            resume=args.resume,steps=200)
        histories[name]=history;nets[name]=net
        append_event(events,'matched_field_trained',source=name,rows=len(training),
            checkpoint_sha256=digest(out/f'{name}.pt'),steps=200)
    # Future-stage coordinates enter only after both fields have finished fitting.
    rng=np.random.default_rng(plan['donor_seed'])
    donor_all=np.flatnonzero(stages==8.)
    donor_local=np.sort(rng.choice(donor_all,1500,replace=False))
    np.save(out/'donor_coordinate_rows.npy',source_rows[donor_local])
    donor=z[donor_local]
    core,_=load_core();results=[]
    for target in [8.25,8.5]:
        target_all=np.flatnonzero(stages==target)
        target_local=np.sort(np.random.default_rng(plan['donor_seed']+int(target*100)).choice(target_all,1000,replace=False))
        np.save(out/f'target_coordinate_rows_{target}.npy',source_rows[target_local])
        truth=z[target_local]
        scores={'persistence':core.mmd_unbiased(donor,truth,n=1000,n_pc=8,seed=plan['donor_seed'])}
        for name,net in nets.items():
            with torch.no_grad():
                predicted=net.trajectory(torch.tensor(donor),[0.,target-8.],step=.125)[-1].numpy()
            scores[name]=core.mmd_unbiased(predicted,truth,n=1000,n_pc=8,seed=plan['donor_seed'])
        result={'cutoff':8.,'target':target,'metric':'8D source latent unbiased multi-kernel MMD; lower is better',
                'raw_mmd':scores,'donor_count':len(donor),'truth_count':len(truth)}
        results.append(result);append_event(events,'source_temporal_pilot_scored',**result)
    passed=all(r['raw_mmd']['full']<min(r['raw_mmd']['sampled'],r['raw_mmd']['persistence']) for r in results)
    report={'status':'completed','plan_sha256':digest(out/'plan.json'),
            'coordinate_report_sha256':plan['coordinate_report_sha256'],
            'checkpoints':{name:digest(out/f'{name}.pt') for name in nets},
            'training_rows':{name:len(fits[name][0]) for name in fits},
            'history':histories,'targets':results,'full_coverage_fit':False,
            'promotion_rule_passed':bool(passed),'official_score':None,
            'conclusion':'Pilot allows full-coverage follow-up only; no challenge claim.' if passed else 'Pilot failed predeclared two-target filter; do not fit full-coverage flow from this exact setup.'}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    public={'updated_utc':now(),'scope':plan['evaluation'],'plan_sha256':report['plan_sha256'],
            'report_sha256':digest(out/'report.json'),'training_rows':report['training_rows'],
            'targets':results,'promotion_rule_passed':bool(passed),
            'full_coverage_fit':False,'challenge_score':None,'official_score':None,'submissions_used':0}
    (HERE/'FULL_SOURCE_FLOW_PILOT_RESULTS.json').write_text(json.dumps(public,indent=2))
    state=json.loads(state_path.read_text())
    state['full_source_flow_pilot_job'].update(status='completed',report_sha256=public['report_sha256'],
        results_report='FULL_SOURCE_FLOW_PILOT_RESULTS.json',promotion_rule_passed=bool(passed))
    state['active_jobs']=[];state['local_process_running']=False
    state_path.write_text(json.dumps(state,indent=2))
    append_event(events,'pilot_completed',promotion_rule_passed=bool(passed))


if __name__=='__main__':
    torch.set_num_threads(2)
    try:
        with threadpool_limits(limits=2):main()
    except Exception as exc:
        folder=HERE/'private/full_source_flow_pilot_01'
        if folder.exists():
            (folder/'failure.json').write_text(json.dumps({'status':'execution_failed','type':type(exc).__name__,'error':str(exc),
                'resume_command':'outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/full_source_flow_pilot.py --resume'},indent=2))
            append_event(folder/'events.jsonl','execution_failed',type=type(exc).__name__,error=str(exc))
        raise
