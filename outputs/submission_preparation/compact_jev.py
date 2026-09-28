"""Offline compact routing packets; no networking, secrets or scientific authority."""
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
AUDIT=HERE.parent/'research_workflow'
MODEL='jev-1.13.0'
def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)

def cache_key(packet):
    return hashlib.sha256(canonical({'schema':'routing-v1','packet':packet}).encode()).hexdigest()

def validate_answers(questions,answers):
    """Fail closed on missing/extra keys and invalid typed values, including NaN."""
    if not questions or not isinstance(answers,dict) or set(answers)!=set(questions):
        raise ValueError('Incomplete answer set; no decision may be cached or promoted.')
    for key,q in questions.items():
        a=answers[key]
        if not isinstance(a,dict):
            raise ValueError('Invalid answer object')
        kind=q['type']
        if kind=='choice':
            if a.get('choice') not in q['criteria']:
                raise ValueError('Unknown choice')
            c=a.get('confidence')
            if isinstance(c,bool) or not isinstance(c,(int,float)) or not math.isfinite(c) or not 0<=c<=1:
                raise ValueError('Missing/invalid confidence; escalate')
        elif kind=='noul':
            n=a.get('noul')
            if isinstance(n,bool) or not isinstance(n,(int,float)) or not math.isfinite(n) or not 0<=n<=1:
                raise ValueError('Invalid noul')
        else:
            raise ValueError('Unsupported routing primitive')
    return answers

def build(phase,record):
    issues=[]
    loop={'loop1':1,'loop2':2}.get(phase)
    if loop:
        rows=json.loads((AUDIT/f'critics/issues_{loop}.json').read_text(encoding='utf-8'))
        issues=[{'id':r['id'],'severity':r['severity'],'owner':r.get('owner',r.get('expert')),'disposition':r['status']} for r in rows]
    state={'phase':phase,'biological_experiments':0,'data_present':False,'original_suite':{'passed':15,'failed':1},
           'issues':issues,'critic_loops_completed_at_decision':loop or 0,
           'scope':'research audit/prospective protocol; no prediction or eligibility claim',
           'facts_version':'2026-09-28-audit-final','review_required':'Scientific interpretation uses full linked artifacts, never this routing summary.'}
    packet={'model':MODEL,'state':state,'questions':record['input']['questions']}
    if len(canonical(packet).encode())>6000:
        raise ValueError('Compact byte budget exceeded; explicitly revise scope, never silently truncate evidence.')
    deterministic={'inventory':'expert_review','loop1':'expert_correction','loop2':'final_risk_review'}[phase]
    return {'phase':phase,'payload':packet,'semantic_cache_key':cache_key(packet),
            'original_payload_bytes':len(canonical(record['input']).encode()),
            'compact_payload_bytes':len(canonical(packet).encode()),
            'historical_actual_input_tokens':record['usage']['input_tokens'],
            'optimized_actual_tokens':None,
            'live_call_needed_for_current_known_state':False,
            'deterministic_action':deterministic,
            'reason':'Known executed counts and explicit workflow phase resolve these questions without a model.',
            'scope':'Counterfactual routing template only. Does not reproduce substantive critic judgments or contain a new Jev answer.'}

def main():
    rows=[build(p,json.loads((AUDIT/f'jev_{p}.json').read_text(encoding='utf-8'))) for p in ('inventory','loop1','loop2')]
    summary={'mode':'offline_only','additional_live_requests':0,'max_live_requests_without_new_approval':0,
             'request_byte_cap':6000,'pin':MODEL,
             'original_total_payload_bytes':sum(r['original_payload_bytes'] for r in rows),
             'compact_total_payload_bytes':sum(r['compact_payload_bytes'] for r in rows),
             'actual_new_token_usage':0,
             'note':'Byte reduction is measured locally, not tokenizer-measured savings or billing. Templates require scope review before any live reuse.'}
    summary['payload_byte_reduction_fraction']=1-summary['compact_total_payload_bytes']/summary['original_total_payload_bytes']
    (HERE/'compact_request_templates.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (HERE/'token_plan.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
