"""Matched OLS stability comparison using only frozen past capture summaries."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now,append_event
from capture_module_estimator import fit,controls

RUN='matched_capture_module_OLS_omission_audit_01'
PUBLIC='MATCHED_MODULE_OLS_OMISSION_AUDIT_RESULTS.json'


def summarize(full,omitted):
    stable=full!=0;comparisons=[]
    for value in omitted:
        agree=np.sign(value)==np.sign(full);stable&=agree
        denominator=float(np.linalg.norm(value)*np.linalg.norm(full))
        comparisons.append({'cosine':float(value@full/denominator) if denominator else None,
             'nonzero_full_sign_agreement':float(agree[full!=0].mean()) if np.any(full!=0) else None,
             'relative_slope_difference':float(np.linalg.norm(value-full)/np.linalg.norm(full)) if np.linalg.norm(full) else None})
    energy=float(full@full)
    return {'stable_nonzero_genes':int(stable.sum()),'stable_shift_energy_fraction':float(full[stable]@full[stable]/energy) if energy else None,
            'full_shift_energy':energy,'omissions':comparisons},stable


def main():
    out=HERE/'private'/RUN
    if out.exists():raise ValueError('Preserve unique frozen run')
    spec_path=HERE/'NEXT_MATCHED_MODULE_OLS_OMISSION_AUDIT.json';spec=json.loads(spec_path.read_text())
    source=HERE/spec['source_run'];report=json.loads((source/'report.json').read_text())
    if digest(source/'report.json')!=spec['source_report_sha256']:raise ValueError('Original report changed')
    if digest(HERE/'capture_module_estimator.py')!=report['plan']['code_sha256']['capture_module_estimator.py']:raise ValueError('Estimator changed')
    plan={**spec,'created_utc':now(),'spec_sha256':digest(spec_path),'source_plan_sha256':digest(source/'plan.json'),
          'code_sha256':{n:digest(HERE/n) for n in ['matched_module_ols_omission_audit.py','capture_module_estimator.py']}}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2));(out/'executed_source.py').write_bytes((HERE/'matched_module_ols_omission_audit.py').read_bytes())
    events=out/'events.jsonl';append_event(events,'plan_frozen',sha256=digest(out/'plan.json'));controls();append_event(events,'numerical_controls_passed');results=[]
    for case in report['results']:
        record={'cutoff':case['cutoff'],'lineage':case['lineage'],'status':case['status'],'stage_capture_counts':case['stage_capture_counts'],
                'every_omission_supported':case['every_omission_supported'],'headline_score':None,'raw_metrics':None,'skills':None,'reward_delta':0}
        if 'artifact_sha256' not in case:results.append(record);continue
        artifact=source/f"cutoff_{case['cutoff']}"/(case['lineage']+'.npz')
        if digest(artifact)!=case['artifact_sha256']:raise ValueError('Frozen capture artifact changed')
        with np.load(artifact) as saved:
            means=saved['capture_means'];stages=saved['capture_stages'];projected,ols,diagnostic=fit(means,stages,case['cutoff'],spec['rank'])
            if not np.array_equal(projected,saved['projected']) or not np.array_equal(ols,saved['unprojected_ols']):raise ValueError('Full slope exact replay failed')
            omitted_p=[];omitted_o=[];omissions=[]
            for i,old in enumerate(case['omissions']):
                if old['status']!='supported':omissions.append(old);continue
                keep=np.arange(len(stages))!=i;op,oo,od=fit(means[keep],stages[keep],case['cutoff'],spec['rank'])
                if not np.array_equal(op,saved['omission_slopes'][len(omitted_p)]):raise ValueError('Omitted projected slope exact replay failed')
                omitted_p.append(op);omitted_o.append(oo);omissions.append({'capture':old['capture'],'stage':old['stage'],'status':'supported'})
            record.update(exact_full_array_replay=True,exact_omission_array_replays=len(omitted_p),source_artifact_sha256=case['artifact_sha256'],omission_eligibility=omissions,
                 projected_retained_OLS_energy=diagnostic['projected_energy']/diagnostic['unprojected_energy'] if diagnostic['unprojected_energy'] else None,
                 discarded_OLS_energy=float((ols-projected)@(ols-projected)))
            if case['every_omission_supported']:
                ps,pm=summarize(projected,omitted_p);os,om=summarize(ols,omitted_o)
                if not np.array_equal(pm,saved['stable_mask']) or ps['stable_shift_energy_fraction']!=case['stable_shift_energy_fraction']:raise ValueError('Original projected stability replay failed')
                record.update(projected_stability=ps,OLS_stability=os,
                              stable_energy_fraction_difference=ps['stable_shift_energy_fraction']-os['stable_shift_energy_fraction'])
            else:record.update(projected_stability=None,OLS_stability=None,stable_energy_fraction_difference=None)
            target=out/f"cutoff_{case['cutoff']}";target.mkdir(exist_ok=True);path=target/(case['lineage']+'.npz')
            np.savez_compressed(path,columns=saved['columns'],projected=projected,unprojected_OLS=ols,omitted_projected=np.array(omitted_p),omitted_OLS=np.array(omitted_o))
            record['matched_artifact_sha256']=digest(path)
        results.append(record);append_event(events,'matched_case_completed',cutoff=case['cutoff'],lineage=case['lineage'],every_omission_supported=case['every_omission_supported'])
    result={'status':'completed','plan':plan,'results':results,'expression_read':False,'target_expression_read':False,'full_panel_scoring_executed':False,'reward_delta':0,'passing_candidates':None}
    (out/'report.json').write_text(json.dumps(result,indent=2));(HERE/PUBLIC).write_text(json.dumps({**result,'report_sha256':digest(out/'report.json')},indent=2));append_event(events,'audit_completed',report_sha256=digest(out/'report.json'))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
