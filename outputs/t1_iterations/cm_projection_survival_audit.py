"""Replay forecast-only diagnostics; no future-expression or scorer execution."""
import json
from collections import Counter
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now,append_event
from lineage_residual_screen import lineage
from cnf_manifold_flow import DensityFlowNet
from cm_observation_calibration import CMObservationForecast
from cm_anchor_support_projection_fullpanel import initialize,project
from projection_survival_diagnostics import alignment,rank_changes,tangent,PastReadGuard
from temporary_forecast_cache import TemporaryForecastCache
from cm_capture_replication_fullpanel import checkpoint
from test_projection_survival_diagnostics import main as diagnostic_controls
from test_joint_margin_solver import main as solver_controls

RUN='cm_projection_survival_audit_01'
PUBLIC='CM_PROJECTION_SURVIVAL_AUDIT_RESULTS.json'


def main():
    out=HERE/'private'/RUN
    if out.exists():raise ValueError('Preserve frozen audit; no automatic retry')
    spec=json.loads((HERE/'NEXT_PROJECTION_SURVIVAL_AUDIT.json').read_text())
    source=HERE/'private/associated_prepared_01';prepared=json.loads((source/'report.json').read_text())
    # Integrity hashing reads opaque bytes; model expression reads are guarded below.
    for name,key in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(source/name)!=prepared[key]:raise ValueError('Prepared input changed: '+name)
    dependencies=['cm_projection_survival_audit.py','projection_survival_diagnostics.py','test_projection_survival_diagnostics.py','test_joint_margin_solver.py','joint_margin_solver.py','cm_anchor_support_projection_fullpanel.py','cm_observation_calibration.py']
    archives={};hashes={}
    for condition in spec['conditions']:
        root=HERE/'private'/condition['archive'];folder=root/f"cutoff_{condition['cutoff']}"
        old=archives.setdefault(condition['archive'],json.loads((root/'report.json').read_text()))
        if old['status']!='completed':raise ValueError('Incomplete archive')
        for name,sha in old['plan']['source_sha256'].items():
            if digest(HERE/name)!=sha:raise ValueError('Archived dependency changed: '+name)
        for name in ['encoder.npz','training.pt','donor_rows.npy','observation_offsets.npz']:
            hashes[str((folder/name).relative_to(HERE))]=digest(folder/name)
        hashes[str((root/'report.json').relative_to(HERE))]=digest(root/'report.json')
    scored={}
    for name in ['cm_anchor_support_projection_fullpanel_01','cm_anchor_support_shrinkage_fullpanel_01']:
        path=HERE/'private'/name/'report.json';scored[name]=json.loads(path.read_text());hashes[str(path.relative_to(HERE))]=digest(path)
    if hashes['private/cm_anchor_support_shrinkage_fullpanel_01/report.json']!=spec['parent_report_sha256']:raise ValueError('Parent changed')
    plan={**spec,'created_utc':now(),'source_sha256':{name:digest(HERE/name) for name in dependencies},
          'predeclaration_sha256':digest(HERE/'NEXT_PROJECTION_SURVIVAL_AUDIT.json'),
          'prepared_report_sha256':digest(source/'report.json'),'archive_artifact_sha256':hashes}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2));(out/'executed_source.py').write_bytes((HERE/'cm_projection_survival_audit.py').read_bytes())
    events=out/'events.jsonl';append_event(events,'plan_frozen',sha256=digest(out/'plan.json'))
    checkpoint(active_jobs=[RUN],local_process_running=True,active_run_path='private/'+RUN,cm_projection_survival_job={'status':'running'})
    diagnostic_controls();solver_controls(adaptive=True);append_event(events,'numerical_controls_passed')
    metadata=pd.read_csv(source/'selected_metadata.csv');stages=metadata.numeric_stage.to_numpy(float);samples=metadata['sample'].astype(str).to_numpy()
    labels=metadata.celltype_extended_atlas.map(lineage).to_numpy()
    panel_path=HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt';panel=panel_path.read_text().splitlines()
    symbols=pd.read_csv(source/'genes.csv').symbol.fillna('').tolist();counts=Counter(symbols)
    lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
    mapped=np.array([i for i,s in enumerate(panel) if s in lookup]);atlas=np.array([lookup[panel[i]] for i in mapped])
    for old in archives.values():
        if digest(panel_path)!=old['plan']['panel_sha256']:raise ValueError('Archived full panel changed')
    expression=np.load(source/'expression.npy',mmap_mode='r');results=[]
    for condition in spec['conditions']:
        cutoff,target,held=condition['cutoff'],condition['target'],condition['held_capture']
        archive=HERE/'private'/condition['archive']/f'cutoff_{cutoff}'
        old=next(f for f in archives[condition['archive']]['folds'] if f['cutoff']==cutoff)
        if old['held_capture']!=held or old['target']!=target:raise ValueError('Archived condition changed')
        donors_rows=np.load(archive/'donor_rows.npy');encoded=dict(np.load(archive/'encoder.npz'));past=encoded['past_rows'];guard=encoded['guard_features']
        if digest(archive/'donor_rows.npy')!=old['donor_rows_sha256'] or np.any(samples[past]==held) or np.any(stages[past]>cutoff):raise ValueError('Past split changed')
        x=PastReadGuard(expression,stages,cutoff)
        donors=np.zeros((len(donors_rows),len(panel)),np.float32);donors[:,mapped]=x[np.ix_(donors_rows,atlas)]
        fit_stages=stages.copy();fit_stages[samples==held]=np.inf
        net=DensityFlowNet(encoded['basis'],encoded['pca_center'],cutoff,float(stages[past].min())-.25)
        net.load_state_dict(torch.load(archive/'training.pt',weights_only=False,map_location='cpu')['net']);net.eval()
        program=CMObservationForecast(x,fit_stages,cutoff,donors,panel,symbols,net,encoded['center'],encoded['scale'],encoded['features'],guard)
        program.configure(.5,.5);mask=labels[donors_rows]=='cardiomyocyte';offsets=dict(np.load(archive/'observation_offsets.npz'))
        generation={};cache=TemporaryForecastCache(HERE/'private/temporary_cache')
        try:
            for name,alpha,scope in [('anchor_unshrunk',0.,'none'),('cm_025',.25,'cm')]:
                program.set_observation_calibration(offsets['cm' if scope=='cm' else 'global'],mask if scope=='cm' else np.ones(len(donors),bool),alpha,scope)
                pred,_,audit=program.predict(target,'joint',1.,sampling='systematic')
                sha=cache.put(name,pred);generation[name]={'prediction_sha256':sha}
                if sha!=old['generation'][name]['prediction_sha256']:raise ValueError('Original forecast replay changed: '+name)
                del pred
            with cache.read('anchor_unshrunk',consume=False) as anchor,cache.read('cm_025',consume=False) as cm:
                reference=anchor[np.ix_(mask,mapped)].astype(float)
                initial_full=initialize(cm,anchor,mask,mapped)
                identity,identity_audit=project(anchor,anchor,donors,mask,mapped,guard)
                if identity is None or cache.put('identity',identity)!=generation['anchor_unshrunk']['prediction_sha256']:raise ValueError('Identity failed')
                del identity
                displacement={};diagnostics={}
                for label,beta,archive_name,initial_name,projected_name in [
                    ('full',1.,'cm_anchor_support_projection_fullpanel_01','anchor_support_initial','anchor_support_projected'),
                    ('quarter',.25,'cm_anchor_support_shrinkage_fullpanel_01','anchor_support_initial_025','anchor_support_projected_025')]:
                    initial=initial_full.copy()
                    initial[np.ix_(mask,mapped)]=(reference+beta*(initial_full[np.ix_(mask,mapped)].astype(float)-reference)).astype(np.float32)
                    scored_fold=next(f for f in scored[archive_name]['folds'] if f['cutoff']==cutoff and f['held_capture']==held)
                    initial_sha=cache.put(label+'_initial',initial)
                    if initial_sha!=scored_fold['generation'][initial_name]['prediction_sha256']:raise ValueError('Initialization replay mismatch')
                    projected,audit=project(initial,anchor,donors,mask,mapped,guard)
                    if projected is None:raise ValueError('Previously valid projection failed')
                    projected_sha=cache.put(label+'_projected',projected)
                    if projected_sha!=scored_fold['generation'][projected_name]['prediction_sha256']:raise ValueError('Projection replay mismatch')
                    proposed=initial[np.ix_(mask,mapped)].astype(float)-reference
                    surviving=projected[np.ix_(mask,mapped)].astype(float)-reference
                    tangent_value,tangent_audit=tangent(reference,proposed)
                    if not tangent_audit['finite']:raise ValueError('Nonfinite tangent diagnostic')
                    diagnostics[label]={**alignment(proposed,surviving),'anchor_l2':float(np.linalg.norm(reference)),
                        'proposed_relative_to_anchor':float(np.linalg.norm(proposed)/np.linalg.norm(reference)),
                        'surviving_relative_to_anchor':float(np.linalg.norm(surviving)/np.linalg.norm(reference)),
                        'rank_changes_initial':rank_changes(reference,reference+proposed),
                        'rank_changes_projected':rank_changes(reference,reference+surviving),'tangent':tangent_audit,'projection':audit}
                    generation[label]={'initial_sha256':initial_sha,'projected_sha256':projected_sha}
                    displacement[label]=(proposed,surviving)
                    del initial,projected,tangent_value
                initial_difference=displacement['full'][0]-displacement['quarter'][0];projected_difference=displacement['full'][1]-displacement['quarter'][1]
                sensitivity={'initial_l2_difference':float(np.linalg.norm(initial_difference)),
                    'projected_l2_difference':float(np.linalg.norm(projected_difference)),
                    'projected_max_absolute_difference':float(np.max(np.abs(projected_difference))),
                    'response_ratio':float(np.linalg.norm(projected_difference)/np.linalg.norm(initial_difference))}
                del initial_full,reference,displacement,initial_difference,projected_difference
            result={**condition,'observed_cm_cells':int(mask.sum()),'generation':generation,'diagnostics':diagnostics,
                    'amplitude_sensitivity':sensitivity,'future_expression_read':False,'expression_reads':x.reads,'max_expression_stage_read':x.max_stage,
                    'headline_score':None,'raw_metrics':None,'skills':None,'reward_delta':0}
            results.append(result);append_event(events,'condition_diagnosed',**result)
            (out/'report.partial.json').write_text(json.dumps({'status':'running','plan':plan,'results':results},indent=2))
        finally:cache.close()
        del program,net,encoded,donors,x
    report={'status':'completed','plan':plan,'results':results,'all_replays_valid':True,'future_expression_read':False,
            'headline_score':None,'raw_metrics':None,'skills':None,'reward_delta':0,'official_score':None}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    public={**report,'plan_sha256':digest(out/'plan.json'),'report_sha256':digest(out/'report.json')}
    (HERE/PUBLIC).write_text(json.dumps(public,indent=2))
    checkpoint(active_jobs=[],local_process_running=False,active_run_path=None,cm_projection_survival_job={'status':'completed','report_sha256':public['report_sha256'],'metrics_available':False})
    append_event(events,'audit_completed',all_replays_valid=True,future_expression_read=False,reward_delta=0)


if __name__=='__main__':
    torch.set_num_threads(2)
    try:
        with threadpool_limits(limits=2):main()
    except Exception as exc:
        out=HERE/'private'/RUN
        if out.exists():
            append_event(out/'events.jsonl','run_failed',exception_type=type(exc).__name__,message=str(exc))
            checkpoint(active_jobs=[],local_process_running=False,active_run_path=None,cm_projection_survival_job={'status':'failed','error_type':type(exc).__name__,'error':str(exc)})
        raise
