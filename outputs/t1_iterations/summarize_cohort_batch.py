"""Publish full metric vectors and calibration comparisons for completed trials."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old = json.loads((HERE/'private/state_search_01/report.json').read_text())
    baseline = [{k:p[k] for k in ['seed','floor','ceiling']} for p in old['panels']]
    records = {}
    for name in ['balanced_proxy_state_search_01','importance_backtest_01']:
        path = HERE/'private'/name/'report.json'
        r = json.loads(path.read_text())
        same = [{k:p[k] for k in ['seed','floor','ceiling']} for p in r['panels']] == baseline
        if not same: raise ValueError('Calibration panels changed; cannot compare')
        vectors = {}
        for summary in r['summaries']:
            candidate = summary['candidate']
            results = [next(v for v in p['results'] if v['candidate'] == candidate) for p in r['panels']]
            vectors[candidate] = {**summary,'mean_metric_skills_percent':{
                k:100*sum(v['skills'][k] for v in results)/len(results) for k in results[0]['skills']},
                'raw_results':results}
        records[name] = {'report_sha256':sha(path),'identical_calibration_panels':same,
                         'evaluations':sum(len(p['results']) for p in r['panels']),
                         'best_development':r['best_development'],'candidates':vectors,
                         'promoted':False}
    payload = {'runs':records,'decision':'Retain unit_k16; new coverage and importance variants underperform.',
               'gate_72_passed':False,'official_score':None,'submissions_used':0,'jev_requests_used':0}
    (HERE/'COHORT_BATCH_RESULTS.json').write_text(json.dumps(payload,indent=2))
    (HERE/'COHORT_BATCH_RESULTS.md').write_text(
        '# Completed source-cohort batch\n\n'
        'All three seeds and all four floor/ceiling calibration anchors exactly match the previous state-search panels. '
        'The 27-evaluation broad-cohort grid has best mean 51.596 (unit_k8). '
        'The 24-evaluation source-importance batch has best new mean 51.629 (25% adaptation, 8 states); '
        'the replayed incumbent reproduces mean 55.312 exactly. No new variant is promoted.\n\n'
        'Importance sampling uses observed E8.5 donor states and atlas rows <=E8.5 only. It selects unique rows without replacement '
        'and applies the same bounded source-to-anchor state ratio across past stages. This avoids duplicate-cell inflation but can distort '
        'past demographic trends; the resulting sampled composition does not identify biological growth.\n\n'
        'COHORT_BATCH_RESULTS.json retains every candidate, all raw metric results, mean four-metric skill vectors and report hashes. '
        'The local >72 gate remains unmet. No official submission or Jev request was made by these experiments. '
        'The separate user-requested E10.5 progress export is an export override, not research promotion.\n')
    path = HERE/'BALANCED_COHORT_RESULTS.json'
    r = json.loads(path.read_text()); r['development_result'] = records['balanced_proxy_state_search_01']
    path.write_text(json.dumps(r,indent=2))
    path = HERE/'BALANCED_COHORT_RESULTS.md'
    s = path.read_text().replace('Status: full-panel development grid running.',
        'Status: completed; best broad-cohort mean 51.596, below incumbent 55.312. Variant rejected.')
    s = s.replace('Three unchanged full-panel challenge panels will assess all four metrics.',
        'Three unchanged full-panel challenge panels assessed all four metrics; floor/ceiling calibration anchors match exactly.')
    s = s.replace('Compare with prior incumbent only after confirming calibration panels match.',
        'The comparison with the prior incumbent uses verified identical calibration panels.')
    path.write_text(s)
    path = HERE/'LOCAL_OPTIMIZATION_STATE.json'
    state = json.loads(path.read_text())
    state.update(updated_utc=datetime.now(timezone.utc).isoformat(),local_process_running=False,
        resume_command=None,next_experiment='Develop a past-only continuous time and tissue-adaptation mechanism; these source-coverage and importance ablations are rejected.')
    state['cohort_batch_result'] = {'report':'outputs/t1_iterations/COHORT_BATCH_RESULTS.json',
        'report_sha256':sha(HERE/'COHORT_BATCH_RESULTS.json'),'evaluations':51,'promoted':False}
    export = HERE.parents[1]/'outputs/t1_submissions/unit16_progress_20260928_01/report.json'
    if export.exists():
        state['progress_export'] = {'report':str(export.relative_to(HERE.parents[1])).replace('\\','/'),
            'report_sha256':sha(export),'purpose':'User-requested progress; readiness unmet',
            'future_refit_performed':True,'future_submission_exported':True,'uploaded':False,
            'artifact_sha256':json.loads(export.read_text())['format_validation']['sha256']}
    path.write_text(json.dumps(state,indent=2))
    path = HERE/'METRIC_RESEARCH_QUEUE.json'
    queue = json.loads(path.read_text())
    # Locate the existing coverage entry without depending on the queue schema name.
    for value in queue.values():
        if not isinstance(value,list): continue
        for entry in value:
            if not isinstance(entry,dict) or entry.get('id') != 'source_cohort_coverage': continue
            entry.update(status='completed_ablation_rejected',
                result_report='outputs/t1_iterations/COHORT_BATCH_RESULTS.json',
                result_report_sha256=sha(HERE/'COHORT_BATCH_RESULTS.json'),
                next_action='Do not rerun this frozen grid. Investigate continuous temporal dynamics and source adaptation using past-only validation.',
                importance_ablation='Observed E8.5 state-ratio sampling also failed; best new mean 51.629 versus incumbent 55.312.',
                research_family_exhausted=False)
    path.write_text(json.dumps(queue,indent=2))


if __name__ == '__main__': main()
