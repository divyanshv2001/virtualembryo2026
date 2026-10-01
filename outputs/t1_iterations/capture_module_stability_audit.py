"""Past-only capture-module stability audit with every omission explicitly recorded."""
import json
from collections import Counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now,append_event
from lineage_residual_screen import lineage
from projection_survival_diagnostics import PastReadGuard
from capture_module_estimator import fit,controls

RUN='capture_module_stability_audit_01'
PUBLIC='CAPTURE_MODULE_STABILITY_AUDIT_RESULTS.json'


def main():
    out=HERE/'private'/RUN
    if out.exists():raise ValueError('Preserve unique frozen run')
    spec_path=HERE/'NEXT_CAPTURE_MODULE_STABILITY_AUDIT.json';spec=json.loads(spec_path.read_text())
    source=HERE/'private/associated_prepared_01';prepared=json.loads((source/'report.json').read_text())
    for name,key in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(source/name)!=prepared[key]:raise ValueError('Source changed')
    m=pd.read_csv(source/'selected_metadata.csv');stages=m.numeric_stage.to_numpy(float)
    captures=m['sample'].astype(str).to_numpy();labels=m.celltype_extended_atlas.map(lineage).to_numpy()
    symbols=pd.read_csv(source/'genes.csv').symbol.fillna('').tolist();counts=Counter(symbols)
    panel_path=HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt';panel=set(panel_path.read_text().splitlines())
    columns=np.array([i for i,s in enumerate(symbols) if s in panel and counts[s]==1]);x=np.load(source/'expression.npy',mmap_mode='r')
    plan={**spec,'created_utc':now(),'spec_sha256':digest(spec_path),'prepared_report_sha256':digest(source/'report.json'),
          'panel_sha256':digest(panel_path),'mapped_unique_genes':len(columns),
          'code_sha256':{n:digest(HERE/n) for n in ['capture_module_stability_audit.py','capture_module_estimator.py','projection_survival_diagnostics.py','lineage_residual_screen.py']}}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2));(out/'executed_source.py').write_bytes((HERE/'capture_module_stability_audit.py').read_bytes())
    events=out/'events.jsonl';append_event(events,'plan_frozen',sha256=digest(out/'plan.json'));controls();append_event(events,'numerical_controls_passed');results=[]
    for cutoff in spec['cutoffs']:
        guard=PastReadGuard(x,stages,cutoff);levels=[cutoff-.5,cutoff-.25,cutoff];folder=out/f'cutoff_{cutoff}';folder.mkdir()
        for group in sorted(set(labels)):
            entries=[];means=[];split_rows=[]
            for t in levels:
                for cap in sorted(set(captures[stages==t])):
                    rows=np.flatnonzero((stages==t)&(labels==group)&(captures==cap))
                    if len(rows)<spec['min_capture_cells']:continue
                    value=np.zeros(len(columns))
                    for start in range(0,len(columns),256):
                        col=columns[start:start+256];value[start:start+len(col)]=np.asarray(guard[np.ix_(rows,col)],float).mean(0)
                    entries.append({'stage':t,'capture':cap,'cells':len(rows)});means.append(value);split_rows.extend(rows.tolist())
            stage_counts=[sum(e['stage']==t for e in entries) for t in levels]
            record={'cutoff':cutoff,'lineage':group,'stage_capture_counts':stage_counts,'captures':entries,'max_fit_stage':guard.max_stage,
                    'headline_score':None,'raw_metrics':None,'skills':None,'reward_delta':0}
            if min(stage_counts)<spec['min_stage_captures_fit']:
                record.update(status='unsupported_stage_captures',every_omission_supported=False);results.append(record);continue
            means=np.stack(means);capture_stages=np.array([e['stage'] for e in entries])
            try:projected,ols,diagnostic=fit(means,capture_stages,cutoff,spec['rank'])
            except ValueError as exc:
                record.update(status='unsupported_numerical_rank',reason=str(exc),every_omission_supported=False);results.append(record);continue
            omissions=[];stable=projected!=0;omitted_slopes=[]
            for i,entry in enumerate(entries):
                keep=np.arange(len(entries))!=i;remaining=[int(np.sum(capture_stages[keep]==t)) for t in levels]
                omission={'capture':entry['capture'],'stage':entry['stage'],'remaining_stage_counts':remaining}
                if min(remaining)<spec['min_stage_captures_fit']:
                    omission.update(status='unsupported_stage_captures');omissions.append(omission);continue
                try:op,oo,od=fit(means[keep],capture_stages[keep],cutoff,spec['rank'])
                except ValueError as exc:
                    omission.update(status='unsupported_numerical_rank',reason=str(exc));omissions.append(omission);continue
                agree=np.sign(op)==np.sign(projected);stable&=agree;den=float(np.linalg.norm(op)*np.linalg.norm(projected))
                omission.update(status='supported',cosine=float(op@projected/den) if den else None,
                                nonzero_full_sign_agreement=float(agree[projected!=0].mean()) if np.any(projected!=0) else None,
                                relative_projected_slope_difference=float(np.linalg.norm(op-projected)/np.linalg.norm(projected)) if np.linalg.norm(projected) else None,
                                numerical_rank=od['numerical_rank']);omissions.append(omission);omitted_slopes.append(op)
            complete=all(o['status']=='supported' for o in omissions)
            artifact=folder/(group+'.npz');np.savez_compressed(artifact,columns=columns,capture_means=means,capture_stages=capture_stages,
                 fit_rows=np.array(split_rows),projected=projected,unprojected_ols=ols,omission_slopes=np.array(omitted_slopes),
                 stable_mask=stable if complete else np.zeros_like(stable))
            energy=float(projected@projected)
            record.update(status='completed_stability' if complete else 'fit_only_stability_unsupported',every_omission_supported=complete,
                          fit_diagnostic=diagnostic,omissions=omissions,artifact_sha256=digest(artifact),
                          stable_nonzero_genes=int(stable.sum()) if complete else None,
                          stable_shift_energy_fraction=float(projected[stable]@projected[stable]/energy) if complete and energy else None)
            results.append(record);append_event(events,'lineage_audited',cutoff=cutoff,lineage=group,every_omission_supported=complete,artifact_sha256=digest(artifact))
    report={'status':'completed','plan':plan,'results':results,'target_expression_read':False,'full_panel_scoring_executed':False,'reward_delta':0,'passing_candidates':None}
    (out/'report.json').write_text(json.dumps(report,indent=2));(HERE/PUBLIC).write_text(json.dumps({**report,'report_sha256':digest(out/'report.json')},indent=2))
    append_event(events,'audit_completed',report_sha256=digest(out/'report.json'))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
