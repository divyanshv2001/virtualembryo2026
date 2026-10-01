"""Evaluate unchanged archived covariance forecasts across target captures."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now,append_event
from lineage_residual_screen import lineage
from cm_covariance_geometry import balanced_covariance,distance_squared
from cm_capture_replication_fullpanel import checkpoint

RUN='cm_covariance_capture_audit_01'
PUBLIC='CM_COVARIANCE_CAPTURE_AUDIT_RESULTS.json'


def main():
    out=HERE/'private'/RUN
    if out.exists():raise ValueError('Preserve frozen run; no retry')
    spec_path=HERE/'NEXT_CM_COVARIANCE_CAPTURE_AUDIT.json';spec=json.loads(spec_path.read_text())
    parent=HERE/'private/cm_covariance_sensitivity_audit_01'
    if digest(parent/'report.json')!=spec['parent_report_sha256']:raise ValueError('Parent changed')
    old=json.loads((parent/'report.json').read_text())
    for name,sha in old['plan']['source_sha256'].items():
        if digest(HERE/name)!=sha:raise ValueError('Archived dependency changed: '+name)
    source=HERE/'private/associated_prepared_01';prepared=json.loads((source/'report.json').read_text())
    if digest(source/'report.json')!=old['plan']['prepared_report_sha256']:raise ValueError('Prepared report changed')
    for name,key in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(source/name)!=prepared[key]:raise ValueError('Prepared input changed')
    hashes={}
    for case in spec['conditions']:
        folder=parent/case['archive_folder']
        for name in ['basis.npz','covariance_forecasts.npz']:
            hashes[(folder/name).relative_to(HERE).as_posix()]=digest(folder/name)
    plan={**spec,'created_utc':now(),'spec_sha256':digest(spec_path),'archive_hashes':hashes,
          'source_sha256':{name:digest(HERE/name) for name in ['cm_covariance_capture_audit.py','cm_covariance_geometry.py']}}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2));(out/'executed_source.py').write_bytes((HERE/'cm_covariance_capture_audit.py').read_bytes())
    events=out/'events.jsonl';append_event(events,'plan_frozen',sha256=digest(out/'plan.json'))
    checkpoint(active_jobs=[RUN],active_run_path='private/'+RUN,local_process_running=True,cm_covariance_capture_job={'status':'running'})
    metadata=pd.read_csv(source/'selected_metadata.csv');stages=metadata.numeric_stage.to_numpy(float);captures=metadata['sample'].astype(str).to_numpy();labels=metadata.celltype_extended_atlas.map(lineage).to_numpy()
    expression=np.load(source/'expression.npy',mmap_mode='r');results=[]
    for case in spec['conditions']:
        folder=parent/case['archive_folder'];basis=dict(np.load(folder/'basis.npz'));forecasts=dict(np.load(folder/'covariance_forecasts.npz'))
        prior=next(r for r in old['results'] if r['cutoff']==case['cutoff'])
        if np.any(stages[basis['fit_rows']]>case['cutoff']):raise ValueError('Basis future leakage')
        rows_by_capture={held:np.flatnonzero((stages==case['target'])&(captures==held)&(labels=='cardiomyocyte')) for held in case['target_captures']}
        if any(len(rows)<spec['min_capture_cells'] for rows in rows_by_capture.values()):raise ValueError('Target support changed')
        append_event(events,'archived_forecasts_verified_before_target_reads',cutoff=case['cutoff'],target=case['target'],forecast_sha256=digest(folder/'covariance_forecasts.npz'),basis_sha256=digest(folder/'basis.npz'))
        target_covariances=[];records=[]
        for held,rows in rows_by_capture.items():
            raw=np.asarray(expression[np.ix_(rows,basis['features'])],float)
            latent=(((raw-basis['center'])/basis['scale'])-basis['pca_mean'])@basis['components'].T
            covariance,support=balanced_covariance(latent,captures[rows],spec['min_capture_cells']);target_covariances.append(covariance)
            diagnostic=[{'candidate':name,'bures_covariance_distance_squared':distance_squared(value,covariance),
                         'frobenius_covariance_error':float(np.linalg.norm(value-covariance))} for name,value in forecasts.items()]
            if held==prior['held_capture']:
                expected={r['candidate']:r for r in prior['diagnostics']}
                if any(abs(r['bures_covariance_distance_squared']-expected[r['candidate']]['bures_covariance_distance_squared'])>1e-9 or abs(r['frobenius_covariance_error']-expected[r['candidate']]['frobenius_covariance_error'])>1e-9 for r in diagnostic):raise ValueError('Original target diagnostic replay mismatch')
            records.append({'held_capture':held,'cells':len(rows),'support':support,'diagnostics':diagnostic,'original_target_replay':held==prior['held_capture']})
            del raw,latent
        pairwise=[{'capture_a':records[i]['held_capture'],'capture_b':records[j]['held_capture'],'bures_covariance_distance_squared':distance_squared(target_covariances[i],target_covariances[j])}
                  for i in range(len(records)) for j in range(i+1,len(records))]
        summary=[]
        for name in forecasts:
            candidate=[next(r for r in rec['diagnostics'] if r['candidate']==name)['bures_covariance_distance_squared'] for rec in records]
            base=[next(r for r in rec['diagnostics'] if r['candidate']=='persistence_cm')['bures_covariance_distance_squared'] for rec in records]
            summary.append({'candidate':name,'errors':candidate,'paired_difference_vs_persistence':[a-b for a,b in zip(candidate,base)],'captures_better_than_persistence':sum(a<b for a,b in zip(candidate,base))})
        result={**case,'captures':records,'summary':summary,'between_capture_covariance_distances':pairwise,'training_executed':False,'original_target_replay_passed':True,
                'headline_score':None,'raw_metrics':None,'skills':None,'reward_delta':0}
        results.append(result);append_event(events,'stage_diagnosed',**result)
        (out/'report.partial.json').write_text(json.dumps({'status':'running','plan':plan,'results':results},indent=2))
    report={'status':'completed','plan':plan,'results':results,'headline_score':None,'raw_metrics':None,'skills':None,'reward_delta':0,'official_score':None,'full_panel_scoring_executed':False,'training_executed':False}
    (out/'report.json').write_text(json.dumps(report,indent=2));public={**report,'report_sha256':digest(out/'report.json')};(HERE/PUBLIC).write_text(json.dumps(public,indent=2))
    append_event(events,'capture_audit_completed',reward_delta=0)
    checkpoint(active_jobs=[],active_run_path=None,local_process_running=False,cm_covariance_capture_job={'status':'completed','report_sha256':public['report_sha256'],'metrics_available':False})


if __name__=='__main__':
    try:
        with threadpool_limits(limits=2):main()
    except Exception as exc:
        out=HERE/'private'/RUN
        if out.exists():
            append_event(out/'events.jsonl','run_failed',exception_type=type(exc).__name__,message=str(exc))
            checkpoint(active_jobs=[],active_run_path=None,local_process_running=False,cm_covariance_capture_job={'status':'failed','error':str(exc)})
        raise
