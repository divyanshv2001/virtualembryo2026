"""Past-fitted CM covariance forecasts; latent diagnostics, not full-panel scores."""
import json
from collections import Counter
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now,append_event
from lineage_residual_screen import lineage
from projection_survival_diagnostics import PastReadGuard
from cm_covariance_geometry import balanced_covariance,pooled_covariance,forecast,project_psd,distance_squared
from test_cm_covariance_geometry import main as controls
from cm_capture_replication_fullpanel import checkpoint

RUN='cm_covariance_sensitivity_audit_01'
PUBLIC='CM_COVARIANCE_SENSITIVITY_AUDIT_RESULTS.json'


def main():
    out=HERE/'private'/RUN
    if out.exists():raise ValueError('Preserve frozen run; no automatic retry')
    spec_path=HERE/'NEXT_CM_COVARIANCE_SENSITIVITY_AUDIT.json';spec=json.loads(spec_path.read_text())
    source=HERE/'private/associated_prepared_01';prepared=json.loads((source/'report.json').read_text())
    for name,key in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(source/name)!=prepared[key]:raise ValueError('Prepared source changed')
    metadata=pd.read_csv(source/'selected_metadata.csv');stages=metadata.numeric_stage.to_numpy(float);captures=metadata['sample'].astype(str).to_numpy();labels=metadata.celltype_extended_atlas.map(lineage).to_numpy()
    symbols=pd.read_csv(source/'genes.csv').symbol.fillna('').tolist();counts=Counter(symbols)
    panel_path=HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt';panel=set(panel_path.read_text().splitlines())
    eligible=np.array([i for i,s in enumerate(symbols) if s in panel and counts[s]==1])
    expression=np.load(source/'expression.npy',mmap_mode='r')
    plan={**spec,'created_utc':now(),'spec_sha256':digest(spec_path),'prepared_report_sha256':digest(source/'report.json'),'panel_sha256':digest(panel_path),
          'source_sha256':{name:digest(HERE/name) for name in ['cm_covariance_sensitivity_audit.py','cm_covariance_geometry.py','test_cm_covariance_geometry.py','projection_survival_diagnostics.py','lineage_residual_screen.py']}}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2));(out/'executed_source.py').write_bytes((HERE/'cm_covariance_sensitivity_audit.py').read_bytes())
    events=out/'events.jsonl';append_event(events,'plan_frozen',sha256=digest(out/'plan.json'));controls();append_event(events,'numerical_controls_passed')
    checkpoint(active_jobs=[RUN],active_run_path='private/'+RUN,local_process_running=True,cm_covariance_sensitivity_job={'status':'running'})
    results=[]
    for condition in spec['conditions']:
        cutoff,target,held=condition['cutoff'],condition['target'],condition['held_capture'];folder=out/f'cutoff_{cutoff}_capture_{held}';folder.mkdir()
        x=PastReadGuard(expression,stages,cutoff)
        past_rows=np.flatnonzero((stages<=cutoff)&(captures!=held))
        cm_rows=past_rows[labels[past_rows]=='cardiomyocyte']
        times=np.unique(stages[cm_rows]);previous,current=times[-2:]
        if current!=cutoff or min((stages[cm_rows]==previous).sum(),(stages[cm_rows]==current).sum())<20:raise ValueError('Insufficient historical CM stage support')
        cm_all=np.asarray(x[np.ix_(cm_rows,eligible)],float)
        variance=cm_all.var(0);selected=eligible[np.argsort(-variance,kind='stable')[:spec['features']]];del cm_all
        fit=np.asarray(x[np.ix_(cm_rows,selected)],float);center=fit.mean(0);scale=np.maximum(fit.std(0),.1)
        pca=PCA(n_components=spec['dimensions'],svd_solver='full');pca.fit((fit-center)/scale);del fit
        z=pca.transform((np.asarray(x[np.ix_(past_rows,selected)],float)-center)/scale)
        past_times=stages[past_rows];past_labels=labels[past_rows];past_captures=captures[past_rows]
        conditional=[];pooled=[];support=[]
        for time in [previous,current]:
            mask=(past_times==time)&(past_labels=='cardiomyocyte')
            covariance,records=balanced_covariance(z[mask],past_captures[mask],spec['min_capture_cells'])
            conditional.append(covariance);support.append({'time':float(time),'captures':records})
            mask=past_times==time;pooled.append(pooled_covariance(z[mask],past_labels[mask]))
        a,b=conditional;learned,null,audit=forecast(a,b,previous,current,target,spec['seed'])
        factor=(target-current)/(current-previous)
        candidates={'persistence_cm':b,'identity_control':b.copy(),'bures_cm':learned,'bures_shuffle':null,
                    'lw_linear_cm':project_psd(b+factor*(b-a)),
                    'pooled_oas_linear':project_psd(pooled[1]+factor*(pooled[1]-pooled[0]))}
        if not np.array_equal(candidates['persistence_cm'],candidates['identity_control']):raise ValueError('Identity mismatch')
        if any(not np.isfinite(v).all() or np.linalg.eigvalsh(v).min()<=0 for v in candidates.values()):raise ValueError('Invalid covariance forecast')
        # All fitting/selection ends before target expression is read.
        np.savez(folder/'basis.npz',features=selected,center=center,scale=scale,components=pca.components_,pca_mean=pca.mean_,fit_rows=cm_rows)
        np.savez(folder/'covariance_forecasts.npz',**candidates)
        target_rows=np.flatnonzero((stages==target)&(captures==held)&(labels=='cardiomyocyte'))
        if len(target_rows)<spec['min_capture_cells']:raise ValueError('Insufficient target CM capture')
        np.save(folder/'target_rows.npy',target_rows)
        append_event(events,'covariance_forecasts_frozen_before_target_read',cutoff=cutoff,target=target,held_capture=held,basis_sha256=digest(folder/'basis.npz'),forecast_sha256=digest(folder/'covariance_forecasts.npz'),max_fit_stage=x.max_stage)
        observed=pca.transform((np.asarray(expression[np.ix_(target_rows,selected)],float)-center)/scale)
        target_covariance,target_support=balanced_covariance(observed,captures[target_rows],spec['min_capture_cells'])
        diagnostics=[{'candidate':name,'bures_covariance_distance_squared':distance_squared(value,target_covariance),
                      'frobenius_covariance_error':float(np.linalg.norm(value-target_covariance)),
                      'forecast_eigenvalues':np.linalg.eigvalsh(value).tolist(),
                      'headline_score':None,'raw_metrics':None,'skills':None,'interpretation':'Latent covariance surrogate only; not official MMD or CSS'} for name,value in candidates.items()]
        result={**condition,'diagnostics':diagnostics,'support':support,'target_support':target_support,'geometry':audit,'max_fit_expression_stage':x.max_stage,
                'fit_rows_sha256':digest(folder/'basis.npz'),'target_rows_sha256':digest(folder/'target_rows.npy'),
                'future_expression_read_for_evaluation_only':True,'target_cells':len(target_rows),'covariance_dimensions':spec['dimensions'],
                'headline_score':None,'raw_metrics':None,'skills':None,'reward_delta':0,'full_gene_decoder_available':False}
        results.append(result);append_event(events,'condition_diagnosed',**result)
        (out/'report.partial.json').write_text(json.dumps({'status':'running','plan':plan,'results':results},indent=2))
        del x,z,pca,observed
    report={'status':'completed','plan':plan,'results':results,'headline_score':None,'raw_metrics':None,'skills':None,'reward_delta':0,'official_score':None,'full_panel_scoring_executed':False}
    (out/'report.json').write_text(json.dumps(report,indent=2));public={**report,'report_sha256':digest(out/'report.json')};(HERE/PUBLIC).write_text(json.dumps(public,indent=2))
    append_event(events,'audit_completed',reward_delta=0)
    checkpoint(active_jobs=[],active_run_path=None,local_process_running=False,cm_covariance_sensitivity_job={'status':'completed','report_sha256':public['report_sha256'],'metrics_available':False})


if __name__=='__main__':
    torch.set_num_threads(2)
    try:
        with threadpool_limits(limits=2):main()
    except Exception as exc:
        out=HERE/'private'/RUN
        if out.exists():
            append_event(out/'events.jsonl','run_failed',exception_type=type(exc).__name__,message=str(exc))
            checkpoint(active_jobs=[],active_run_path=None,local_process_running=False,cm_covariance_sensitivity_job={'status':'failed','error':str(exc)})
        raise
