"""Past-only archived forecast replay and scope diagnostics; no scorer."""
import json
from collections import Counter
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event
from lineage_residual_screen import lineage
from cnf_manifold_flow import DensityFlowNet
from cm_observation_calibration import CMObservationForecast
from cm_anchor_support_projection_fullpanel import initialize, project
from early_tangent_controls import construct
from projection_survival_diagnostics import PastReadGuard
from temporary_forecast_cache import TemporaryForecastCache
from cm_capture_replication_fullpanel import checkpoint

RUN='early_forecast_scope_audit_01'
PUBLIC='EARLY_FORECAST_SCOPE_AUDIT_RESULTS.json'


def anchor_base(candidate, anchor, mask, mapped):
    result=anchor.copy()
    result[np.ix_(mask,mapped)]=candidate[np.ix_(mask,mapped)]
    return result


def partition(candidate, anchor, mask):
    delta=candidate.astype(float)-anchor.astype(float)
    inside=delta[mask].sum(0)/len(delta)
    outside=delta[~mask].sum(0)/len(delta)
    full=delta.mean(0)
    return {'full_mean_max':float(np.abs(full).max()),
            'cm_contribution_max':float(np.abs(inside).max()),
            'noncm_contribution_max':float(np.abs(outside).max()),
            'decomposition_error':float(np.abs(full-inside-outside).max()),
            'noncm_changed_entries':int(np.count_nonzero(delta[~mask])),
            'cm_changed_entries':int(np.count_nonzero(delta[mask]))}


def main():
    out=HERE/'private'/RUN
    if out.exists():raise ValueError('Preserve prior evidence; no retry')
    spec_path=HERE/'NEXT_EARLY_SCOPE_AUDIT.json';spec=json.loads(spec_path.read_text())
    parent_path=HERE/'private/early_tangent_metric_hindcast_02/report.json'
    if digest(parent_path)!=spec['parent_report_sha256']:raise ValueError('Parent changed')
    parent=json.loads(parent_path.read_text());oldplan=parent['plan']
    for name,sha in oldplan['source_sha256'].items():
        if digest(HERE/name)!=sha:raise ValueError('Archived dependency changed: '+name)
    source=HERE/'private/associated_prepared_01';prepared=json.loads((source/'report.json').read_text())
    if digest(source/'report.json')!=oldplan['prepared_report_sha256']:raise ValueError('Prepared report changed')
    for name,key in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(source/name)!=prepared[key]:raise ValueError('Prepared input changed')
    artifact_hashes={}
    for condition in spec['conditions']:
        archive=HERE/condition['folder']
        for name in ['encoder.npz','training.pt','donor_rows.npy','observation_offsets.npz']:
            artifact_hashes[(archive/name).relative_to(HERE).as_posix()]=digest(archive/name)
    plan={**spec,'created_utc':now(),'spec_sha256':digest(spec_path),'code_sha256':digest(HERE/'forecast_scope_audit.py'),'archive_artifact_sha256':artifact_hashes}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2));(out/'executed_source.py').write_bytes((HERE/'forecast_scope_audit.py').read_bytes())
    events=out/'events.jsonl';append_event(events,'plan_frozen',sha256=digest(out/'plan.json'))
    checkpoint(active_jobs=[RUN],local_process_running=True,active_run_path='private/'+RUN,early_scope_audit_job={'status':'running'})
    metadata=pd.read_csv(source/'selected_metadata.csv');stages=metadata.numeric_stage.to_numpy(float);samples=metadata['sample'].astype(str).to_numpy()
    labels=metadata.celltype_extended_atlas.map(lineage).to_numpy()
    panel_path=HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt'
    if digest(panel_path)!=oldplan['panel_sha256']:raise ValueError('Panel changed')
    panel=panel_path.read_text().splitlines();symbols=pd.read_csv(source/'genes.csv').symbol.fillna('').tolist();counts=Counter(symbols)
    lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
    mapped=np.array([i for i,s in enumerate(panel) if s in lookup]);atlas=np.array([lookup[panel[i]] for i in mapped])
    expression=np.load(source/'expression.npy',mmap_mode='r');results=[]
    for condition in spec['conditions']:
        cutoff,target,held=condition['cutoff'],condition['target'],condition['held_capture'];archive=HERE/condition['folder']
        old=next(f for f in parent['folds'] if f['held_capture']==held and f['cutoff']==cutoff)
        rows=np.load(archive/'donor_rows.npy');encoded=dict(np.load(archive/'encoder.npz'));past=encoded['past_rows'];guard=encoded['guard_features']
        if digest(archive/'donor_rows.npy')!=old['donor_rows_sha256'] or np.any(samples[past]==held) or np.any(stages[past]>cutoff):raise ValueError('Split changed')
        x=PastReadGuard(expression,stages,cutoff);donors=np.zeros((len(rows),len(panel)),np.float32);donors[:,mapped]=x[np.ix_(rows,atlas)]
        fit_stages=stages.copy();fit_stages[samples==held]=np.inf
        net=DensityFlowNet(encoded['basis'],encoded['pca_center'],cutoff,float(stages[past].min())-.25)
        net.load_state_dict(torch.load(archive/'training.pt',weights_only=False,map_location='cpu')['net']);net.eval()
        program=CMObservationForecast(x,fit_stages,cutoff,donors,panel,symbols,net,encoded['center'],encoded['scale'],encoded['features'],guard);program.configure(.5,.5)
        mask=labels[rows]=='cardiomyocyte';offsets=dict(np.load(archive/'observation_offsets.npz'));cache=TemporaryForecastCache(HERE/'private/temporary_cache');generation={}
        try:
            for name,alpha,scope in [('anchor_unshrunk',0.,'none'),('cm_025',.25,'cm')]:
                program.set_observation_calibration(offsets['cm' if scope=='cm' else 'global'],mask if scope=='cm' else np.ones(len(donors),bool),alpha,scope)
                pred,_,audit=program.predict(target,'joint',1.,sampling='systematic');sha=cache.put(name,pred)
                if sha!=old['generation'][name]['prediction_sha256']:raise ValueError('Replay hash changed: '+name)
                generation[name]={'sha256':sha,'backoff':audit['backoff']};del pred
            with cache.read('anchor_unshrunk',consume=False) as anchor,cache.read('cm_025',consume=False) as cm:
                observed=partition(cm,anchor,mask);full=initialize(cm,anchor,mask,mapped)
                blocks,construction=construct(anchor[np.ix_(mask,mapped)],full[np.ix_(mask,mapped)],spec['seed'])
                initial=full.copy();initial[np.ix_(mask,mapped)]=blocks['learned']
                if cache.put('legacy_initial',initial)!=old['generation']['tangent_learned_initial']['prediction_sha256']:raise ValueError('Legacy tangent replay changed')
                projected,audit=project(initial,anchor,donors,mask,mapped,guard)
                if projected is None or cache.put('legacy_projected',projected)!=old['generation']['tangent_learned_projected']['prediction_sha256']:raise ValueError('Legacy projection changed')
                legacy=partition(projected,anchor,mask);del projected
                isolated=anchor_base(initial,anchor,mask,mapped);del initial,full,blocks
                projected,audit=project(isolated,anchor,donors,mask,mapped,guard)
                isolated_sha=cache.put('anchor_base_initial',isolated)
                isolation={'initial':partition(isolated,anchor,mask),'initial_sha256':isolated_sha,'projection':audit,'projected':None,'projected_sha256':None}
                if projected is not None:
                    isolation['projected']=partition(projected,anchor,mask);isolation['projected_sha256']=cache.put('anchor_base_projected',projected)
                    if isolation['projected']['noncm_changed_entries'] or isolation['projected']['full_mean_max']>1e-5:raise ValueError('Anchor-base scope invariant failed')
                del isolated,projected
            result={**condition,'replays_valid':True,'cm_offset':observed,'legacy_tangent':legacy,'anchor_base':isolation,'generation':generation,'future_expression_read':False,'max_expression_stage_read':x.max_stage,'raw_metrics':None,'skills':None,'headline_score':None,'reward_delta':0}
            results.append(result);append_event(events,'condition_diagnosed',**result)
            (out/'report.partial.json').write_text(json.dumps({'status':'running','plan':plan,'results':results},indent=2))
        finally:cache.close()
        del program,net,donors,encoded,x
    report={'status':'completed','plan':plan,'results':results,'future_expression_read':False,'raw_metrics':None,'skills':None,'headline_score':None,'reward_delta':0,'official_score':None}
    (out/'report.json').write_text(json.dumps(report,indent=2));public={**report,'report_sha256':digest(out/'report.json')};(HERE/PUBLIC).write_text(json.dumps(public,indent=2))
    append_event(events,'audit_completed',reward_delta=0)
    checkpoint(active_jobs=[],local_process_running=False,active_run_path=None,early_scope_audit_job={'status':'completed','report_sha256':public['report_sha256'],'metrics_available':False})


if __name__=='__main__':
    torch.set_num_threads(2)
    try:
        with threadpool_limits(limits=2):main()
    except Exception as exc:
        out=HERE/'private'/RUN
        if out.exists():
            append_event(out/'events.jsonl','run_failed',exception_type=type(exc).__name__,message=str(exc))
            checkpoint(active_jobs=[],local_process_running=False,active_run_path=None,early_scope_audit_job={'status':'failed','error':str(exc)})
        raise
