"""Index preserved local outcomes without inventing missing metric information."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def outcomes(value,context=None,pointer=''):
    context=dict(context or {})
    if isinstance(value,dict):
        for key in ['candidate','config','seed','cutoff','target','held_capture','fold','replicate','prediction_sha256']:
            if key in value:context[key]=value[key]
        for key in ['floor','ceiling']:
            if key in value:context[key]=value[key]
        if isinstance(value.get('generation'),dict):
            context['generation']=value['generation']
        if 'local_score' in value:
            if pointer.endswith('/persistence'):
                context['candidate']='persistence'
            generation=context.pop('generation',{})
            entry=generation.get(context.get('candidate',''),{})
            yield {**context,'prediction_sha256':value.get('prediction_sha256',entry.get('prediction_sha256',entry.get('sha256'))),
                'json_pointer':pointer,'local_score':value['local_score'],
                'raw_metrics':value.get('raw_metrics',value.get('raw',context.get('floor') if pointer.endswith('/persistence') else None)),
                'diagnostics':entry.get('audit'),
                'skills':value.get('skills'),'calibration_valid':value.get('calibration_valid'),
                'guards_passed':value.get('guards_passed',entry.get('guards_passed')),
                'diagnostic_invalid_control':value.get('diagnostic_invalid_control'),
                'invalid_calibration_metrics':value.get('invalid_calibration_metrics'),
                'scope':'Local development; not an official leaderboard score'}
        for key,child in value.items():
            if pointer=='' and key=='panels' and isinstance(value.get('folds'),list) and child==value['folds']:
                # The same scored batch can be exposed under both adapter names.
                # Keep canonical folds once without discarding distinct panels.
                continue
            if pointer=='' and key=='results' and isinstance(child,list) and isinstance(value.get('folds'),list):
                # Final summaries can embed identical scored-fold snapshots.
                # Index the canonical folds once; retain nonidentical snapshots.
                child=[{k:v for k,v in item.items() if k!='scored_fold'}
                       if isinstance(item,dict) and item.get('scored_fold') in value['folds'] else item
                       for item in child]
            if isinstance(child,(dict,list)):
                yield from outcomes(child,context,pointer+'/'+str(key))
    elif isinstance(value,list):
        for i,child in enumerate(value):yield from outcomes(child,context,pointer+'/'+str(i))


def main():
    records=[];sources=[];errors=[]
    private=HERE/'private'
    paths=sorted(private.rglob('report.json'))
    paths+=sorted(p for p in private.rglob('report.partial.json') if not (p.parent/'report.json').exists())
    for path in paths:
        try:
            payload=path.read_bytes();report=json.loads(payload)
            sha=hashlib.sha256(payload).hexdigest()
            plan=path.parent/'plan.json'
            plan_record=report.get('plan',{})
            rows=list(outcomes(report,{'seed':plan_record.get('seed')}))
            generation=report.get('generation',{})
            for row in rows:
                entry=generation.get(row.get('candidate',''),{}) if isinstance(generation,dict) else {}
                if not row.get('prediction_sha256'):row['prediction_sha256']=entry.get('prediction_sha256')
                row.update(report_path=path.relative_to(HERE).as_posix(),report_sha256=sha,
                    plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest() if plan.exists() else None,
                    source_sha256=plan_record.get('source_sha256'),
                    scorer_manifest_sha256=report.get('scorer_manifest_sha256'),
                    diagnostics=row.get('diagnostics') or entry.get('audit'),
                    report_status=report.get('status','unspecified'),
                    historical_score_replay=bool(report.get('scores_are_historical_replays',False)))
            records.extend(rows)
            sources.append({'path':path.relative_to(HERE).as_posix(),'sha256':sha,'outcomes_indexed':len(rows)})
        except (ValueError,OSError) as exc:errors.append({'path':str(path),'error':str(exc)})
    # This is a reproducible index; original reports/events remain the source history.
    ledger=HERE/'SCORE_LEDGER.jsonl'
    content=''.join(json.dumps(r,sort_keys=True)+'\n' for r in records)
    if not ledger.exists() or ledger.read_text()!=content:
        temporary=ledger.with_suffix('.jsonl.tmp')
        temporary.write_text(content)
        temporary.replace(ledger)
    manifest={'records':len(records),'sources':sources,'index_errors':errors,
        'limitations':'Null means not present in the original report. Reports without local_score require schema-specific indexing; original reports/events remain preserved. Partial index hashes change as live work progresses.',
        'ledger_sha256':hashlib.sha256((HERE/'SCORE_LEDGER.jsonl').read_bytes()).hexdigest()}
    (HERE/'SCORE_LEDGER_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'records':len(records),'reports':len(sources),'errors':len(errors)}))
    if errors:raise ValueError('Some reports could not be indexed')


if __name__=='__main__':main()
