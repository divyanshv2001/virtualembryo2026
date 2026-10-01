"""Frozen scope-controlled forecasts, then unchanged full-panel evaluation."""
import json
import numpy as np
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import append_event
from offline_backtest import load_core, Panel
from early_tangent_controls import construct
from forecast_scope_audit import anchor_base, partition
from cm_anchor_support_projection_fullpanel import initialize, project

NAMES=['copy','anchor_unshrunk','anchor_identity','legacy_tangent','isolated_learned','isolated_shuffle']


def score_condition(cache,anchor,cm,donors,mask,mapped,guard,old,expression,stages,archive,events,cutoff,target,held,seed):
    """Called only by the registered scored driver, never the scoreless audit."""
    cache.put('copy',donors);cache.put('anchor_identity',anchor)
    full=initialize(cm,anchor,mask,mapped)
    blocks,construction=construct(anchor[np.ix_(mask,mapped)],full[np.ix_(mask,mapped)],seed)
    generation={};norms={}
    for kind in ['learned','shuffle']:
        initial=anchor.copy();initial[np.ix_(mask,mapped)]=blocks[kind]
        candidate,audit=project(initial,anchor,donors,mask,mapped,guard)
        name='isolated_'+kind
        if candidate is None:
            generation[name]={'valid':False,'audit':audit,'sha256':None}
        else:
            scope=partition(candidate,anchor,mask)
            if scope['noncm_changed_entries'] or scope['full_mean_max']>1e-5:raise ValueError('Whole-panel scope violated')
            norms[kind]=float(np.linalg.norm(candidate[np.ix_(mask,mapped)].astype(float)-anchor[np.ix_(mask,mapped)].astype(float)))
            generation[name]={'valid':True,'audit':audit,'scope':scope,'sha256':cache.put(name,candidate)}
        del initial,candidate
    del full,blocks
    ratio=norms.get('learned',0)/norms['shuffle'] if norms.get('shuffle',0)>0 else None
    negative={'valid':ratio is not None and .9<=ratio<=1.1,'ratio':ratio,'norms':norms,'single_seed':seed}
    # The caller already froze and hash-verified legacy_projected.
    append_event(events,'all_scored_forecasts_frozen_before_target_read',cutoff=cutoff,held_capture=held,generation=generation,negative_control=negative)
    target_rows=np.load(archive/'target_rows.npy')
    if digest(archive/'target_rows.npy')!=old['target_rows_sha256'] or np.any(stages[target_rows]!=target):raise ValueError('Target split changed')
    source=HERE/'private/associated_prepared_01';symbols=__import__('pandas').read_csv(source/'genes.csv').symbol.fillna('').tolist()
    from collections import Counter
    counts=Counter(symbols);lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
    panel=(HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
    atlas=np.array([lookup[panel[i]] for i in mapped])
    future=np.zeros((len(target_rows),len(panel)),np.float32);future[:,mapped]=expression[np.ix_(target_rows,atlas)]
    order=np.random.default_rng(seed).permutation(len(future));core,_=load_core();evaluator=Panel(core,future[order[:1000]],donors,seed)
    floor=evaluator.metrics(donors);ceiling=evaluator.metrics(future[order[1000:]])
    results=[]
    for name in NAMES:
        key='legacy_projected' if name=='legacy_tangent' else name
        if name.startswith('isolated_') and not generation[name]['valid']:
            result={'candidate':name,'raw_metrics':dict.fromkeys(['de_score','de_direction','mmd_u','variogram']),'skills':dict.fromkeys(['de_score','de_direction','mmd_u','variogram']),'local_score':None,'calibration_valid':False,'invalid_reason':generation[name]['audit'].get('reason','Invalid projection')}
        else:
            with cache.read(key,consume=False) as pred:
                raw=evaluator.metrics(pred);result={'candidate':name,'raw_metrics':raw,**evaluator.aggregate(raw,floor,ceiling)}
            if name in ['copy','anchor_unshrunk','anchor_identity','legacy_tangent']:
                oldname={'anchor_identity':'anchor_unshrunk','legacy_tangent':'tangent_learned_projected'}.get(name,name)
                prior=next(r for r in old['results'] if r['candidate']==oldname)
                if abs(result['local_score']-prior['local_score'])>1e-9 or any(abs(raw[m]-prior['raw_metrics'][m])>1e-10 for m in raw):raise ValueError('Scored control replay mismatch: '+name)
        results.append(result);append_event(events,'candidate_scored',cutoff=cutoff,held_capture=held,**result)
    return {'cutoff':cutoff,'target':target,'held_capture':held,'floor':floor,'ceiling':ceiling,'results':results,'generation':generation,'negative_control':negative,'construction':construction,'target_rows_sha256':old['target_rows_sha256'],'evaluation_role':'Previously evaluated target; scope diagnostic, not independent confirmation'}


def finalize(out,report,public_path):
    from metric_critique_reward import assess,total_reward
    from cm_capture_replication_fullpanel import checkpoint
    folds=[r['scored_fold'] for r in report['results']]
    summary=[]
    for name in NAMES:
        rows=[next(r for r in f['results'] if r['candidate']==name) for f in folds]
        summary.append({'candidate':name,'scores':[r['local_score'] for r in rows],'raw_metrics':[r['raw_metrics'] for r in rows],'skills':[r['skills'] for r in rows],'all_calibrations_valid':all(r['calibration_valid'] for r in rows)})
    candidate=next(r for r in summary if r['candidate']=='isolated_learned');control=next(r for r in summary if r['candidate']=='anchor_unshrunk')
    assessment=assess(candidate['skills'],control['skills'],eligible=candidate['all_calibrations_valid'] and control['all_calibrations_valid'] and all(f['negative_control']['valid'] for f in folds))
    report.update(folds=folds,summary=summary,passing_candidates=[],critic_assessments=[assessment],reward_delta=assessment['reward_delta'],promotion_limitation='Reused diagnostic targets; no independent promotion or readiness claim')
    (out/'report.json').write_text(json.dumps(report,indent=2));sha=digest(out/'report.json');public={**report,'report_sha256':sha};public_path.write_text(json.dumps(public,indent=2))
    ledger=HERE/'METRIC_CRITIQUE_REWARD_LEDGER.jsonl';previous=[json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    event={'experiment_id':out.name+'/isolated_learned','assessment':assessment,'plan_sha256':digest(out/'plan.json'),'report_sha256':sha}
    if event['experiment_id'] in {x['experiment_id'] for x in previous}:raise ValueError('Duplicate reward event')
    with ledger.open('a') as handle:handle.write(json.dumps(event)+'\n')
    totals=total_reward(previous+[event]);state=json.loads((HERE/'LOCAL_OPTIMIZATION_STATE.json').read_text());state['metric_critique_reward'].update(current_reward=totals['reward_score'],uncapped_reward=totals['uncapped_reward'])
    checkpoint(metric_critique_reward=state['metric_critique_reward'],isolated_scope_job={'status':'completed','report_sha256':sha,'passing_candidates':[]})
