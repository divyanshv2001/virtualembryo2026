"""Publish complete diagnostic batches and refresh their reproducible score index."""
import argparse
import json
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now
from index_scores import main as index_scores
from summarize_horizon_batch import main as summarize_batches


def finish(run,output,job_key,decision):
    folder=HERE/'private'/run
    if folder.resolve().parent!=(HERE/'private').resolve():raise ValueError('Invalid run path')
    if not output.endswith('.json') or '/' in output or '\\' in output:raise ValueError('Invalid public filename')
    report=json.loads((folder/'report.json').read_text())
    if report.get('status')!='completed':raise ValueError('Batch is not complete')
    if len(report['folds'])!=len(report['plan']['folds']):raise ValueError('Incomplete folds')
    summaries=[]
    for candidate in report['plan']['configs']:
        results=[next(v for v in f['results'] if v['candidate']==candidate) for f in report['folds']]
        valid=all(v['calibration_valid'] for v in results)
        scores=[v['local_score'] for v in results]
        summaries.append({'candidate':candidate,'scores':scores,
            'mean_score':sum(scores)/len(scores) if valid else None,
            'raw_metrics':[v['raw_metrics'] for v in results],
            'skills':[v['skills'] for v in results],'all_calibrations_valid':valid,
            'beats_persistence_on_every_fold':valid and all(s>50 for s in scores)})
    public={'updated_utc':now(),'status':'completed_not_promoted',
        'evaluations':sum(len(f['results']) for f in report['folds']),
        'plan_sha256':digest(folder/'plan.json'),'report_sha256':digest(folder/'report.json'),
        'summaries':summaries,'decision':decision,'scope':report['plan']['scope'],
        'local_72_gate_passed':False,'official_score':None,'submissions_used':0,'jev_requests_used':0}
    (HERE/output).write_text(json.dumps(public,indent=2))
    path=HERE/'LOCAL_OPTIMIZATION_STATE.json';state=json.loads(path.read_text())
    state.setdefault(job_key,{}).update(status='completed_not_promoted',report_sha256=public['report_sha256'],
        results_report=output,evaluations=public['evaluations'],pending_evaluations=0)
    state['local_process_running']=False
    path.write_text(json.dumps(state,indent=2))
    path=HERE/'METRIC_RESEARCH_QUEUE.json';queue=json.loads(path.read_text())
    for entry in queue['paths']:
        if entry.get('run')==run:entry.update(status='implemented_evaluated_not_promoted',results_report=output,decision=decision)
    path.write_text(json.dumps(queue,indent=2))
    index_scores();summarize_batches()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--job-key',required=True)
    parser.add_argument('--decision',required=True);args=parser.parse_args()
    finish(args.run,args.output,args.job_key,args.decision)
