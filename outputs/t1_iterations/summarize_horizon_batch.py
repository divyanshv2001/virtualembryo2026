"""Checkpoint completed horizon audits and publish complete metric vectors."""
import json
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now


def main():
    runs={};running=[]
    for name in ['matched_horizon_audit_01','quantile_horizon_pilot_01','state_quantile_horizon_pilot_01']:
        folder=HERE/'private'/name;path=folder/'report.json'
        if not path.exists():
            running.append(name);continue
        r=json.loads(path.read_text());summaries=[]
        for candidate in r['plan']['configs']:
            results=[next(v for v in f['results'] if v['candidate']==candidate) for f in r['folds']]
            valid=all(v['calibration_valid'] for v in results)
            summaries.append({'candidate':candidate,'all_calibrations_valid':valid,
                'scores':[v['local_score'] for v in results],
                'mean_score':sum(v['local_score'] for v in results)/len(results) if valid else None,
                'mean_skills_percent':{k:100*sum(v['skills'][k] for v in results)/len(results)
                    for k in results[0]['skills']} if valid else None,
                'beats_persistence_on_every_fold':valid and all(v['local_score']>50 for v in results),
                'all_metrics_at_least_50_on_every_fold':valid and all(min(v['skills'].values())>=.5 for v in results)})
        runs[name]={'report_sha256':digest(path),'plan_sha256':digest(folder/'plan.json'),
            'folds':r['folds'],'summaries':summaries,'scope':r['plan']['scope']}
    status='running' if running else 'completed'
    output={'updated_utc':now(),'status':status,'runs':runs,'running':running,
        'local_72_gate_passed':False,'official_score_of_new_model':None,'submissions_used':0,'jev_requests_used':0}
    (HERE/'HORIZON_BATCH_RESULTS.json').write_text(json.dumps(output,indent=2))
    lines=['# One-day source-cohort audits','',
        'Two observed one-day folds, E8.0→E9.0 and E8.25→E9.25, use the same cardiac-associated source cohort, '
        '1,500 donors and complete official gene ordering. Each forecast is frozen before later-stage expression is read for scoring. '
        'Missing/ambiguous genes use zero placeholders in this atlas-only diagnostic; these are not measured challenge genes. '
        'These folds therefore improve horizon matching but do not certify challenge-domain or embryo-independent generalization.','',
        'The prior model scores 47.143 and 47.987. All component ablations also score below persistence on both folds. '
        'This confirms one-day failure within the source cohort; it cannot isolate hidden E10.5 causes.','',
        'The positive-quantile mechanism fits three historical positive-expression quantile distributions, suppresses reversing trends, '
        'shrinks low-support slopes, and preserves the donor zero mask and number of cells. Per-cell mapped mass and protected genes are guarded. '
        'Marginal proposals are monotone; cell-specific mass conservation can change final cross-cell ranks.','',
        'A second pilot conditions those margins on four broad past-fitted states. It retains fixed donor counts, '
        'changes only trusted anchors, requires 20 anchors and 20 source cells at every recent stage, '
        'and applies both within-state and overall covariance guards. It tests composition confounding, not inferred cell proliferation.','',
        'Literature: [Schefzik, Thorarinsdottir and Gneiting (2013)](https://arxiv.org/html/1302.7149v2), '
        'methods 4.1–4.3 and experiments 5.4 read. ECC separates marginal calibration from rank dependence. '
        'Its weather experiments found benefits dependent on the dependence structure. This implementation adapts that separation '
        'to historical positive scRNA margins; it is neither faithful ECC nor evidence that ECC improves this challenge.','',
        '| Run | Candidate | Fold 1 | Fold 2 | Mean |','| --- | --- | ---: | ---: | ---: |']
    for name,r in runs.items():
        for s in r['summaries']:
            if s['mean_score'] is not None:
                a,b=s['scores'];lines.append(f"| {name} | {s['candidate']} | {a:.3f} | {b:.3f} | {s['mean_score']:.3f} |")
    lines += ['',f'Status: {status}. Full four-metric vectors, raw metrics, model audits and provenance hashes are in HORIZON_BATCH_RESULTS.json.',
        'No new prospective export or official submission. The >72 objective remains unfinished.']
    (HERE/'HORIZON_BATCH_RESULTS.md').write_text('\n'.join(lines)+'\n')
    path=HERE/'LOCAL_OPTIMIZATION_STATE.json';state=json.loads(path.read_text())
    state.update(updated_utc=now(),local_process_running=bool(running))
    state['horizon_batch']={'status':status,'running':running,'report':'outputs/t1_iterations/HORIZON_BATCH_RESULTS.json',
        'report_sha256':digest(HERE/'HORIZON_BATCH_RESULTS.json')}
    state['next_experiment']='Finish positive-quantile one-day pilot; advance to challenge development only if it beats persistence on both folds without metric regressions.' if running else 'Implement annotation-conditioned abundance/detection trends on the two declared one-day atlas folds before any challenge refit. Eleven exact challenge labels are shared, but correspondence and changing label granularity need explicit checks. See CHALLENGE_LABEL_ALIGNMENT.json.'
    path.write_text(json.dumps(state,indent=2))
    path=HERE/'METRIC_RESEARCH_QUEUE.json';queue=json.loads(path.read_text())
    entry={'id':'positive_quantile_temporal','metrics':['de_score','de_direction','mmd_u','variogram'],
        'status':'running' if running else 'implemented_evaluated',
        'sources':[{'url':'https://arxiv.org/html/1302.7149v2',
            'read':'Methods sections4.1-4.3 and experiment section5.4; marginal/rank separation and limits.'}],
        'implementation':['positive_quantile_forecast.py','state_quantile_forecast.py'],
        'scope':'Adaptation, not faithful ECC. Three positive-quantile strengths, global and four-state conditional, two one-day associated-cohort folds.',
        'results_report':'outputs/t1_iterations/HORIZON_BATCH_RESULTS.json',
        'results_report_sha256':digest(HERE/'HORIZON_BATCH_RESULTS.json'),
        'research_family_exhausted':False,'submissions_used':0,'jev_requests_used':0,
        'next_action':state['next_experiment']}
    for value in queue.values():
        if isinstance(value,list) and any(isinstance(v,dict) and 'id' in v for v in value):
            previous=next((i for i,v in enumerate(value) if v.get('id')==entry['id']),None)
            if previous is None:value.append(entry)
            else:value[previous]=entry
            break
    path.write_text(json.dumps(queue,indent=2))
    print(json.dumps({'status':status,'running':running,'means':{
        k:{s['candidate']:s['mean_score'] for s in v['summaries']} for k,v in runs.items()}},indent=2))


if __name__=='__main__':main()
