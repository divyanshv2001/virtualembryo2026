"""Shared frozen full-panel scoring; preserve original metrics, splits and gates."""
import numpy as np
from offline_backtest import Panel
from iterate import now
from run_t1 import digest
from scnode_resource_preflight import peak_memory
from tigon_conditional_fullpanel import panel_values, save


def score_frozen_forecasts(core, raw_x, stages, donors, mapped, columns, panel,
                           cutoff, target, plan, names, generation, cache, RUN, report, emit, spec):
    if set(generation)!=set(names) or not all(generation[n].get('prediction_sha256') for n in names):
        raise ValueError('Freeze every named forecast before scoring')
    available=np.flatnonzero(stages==target)
    for seed in plan['scoring_seeds']:
        rows=np.random.default_rng(seed).choice(available,2000,replace=False)
        np.save(RUN/f'target_rows_{seed}.npy',rows)
        future=panel_values(raw_x,rows,mapped,columns,len(panel))
        order=np.random.default_rng(seed).permutation(len(future));np.save(RUN/f'target_order_{seed}.npy',order)
        evaluator=Panel(core,future[order[:1000]],donors,seed)
        floor,ceiling=evaluator.metrics(donors),evaluator.metrics(future[order[1000:]])
        outcomes=[]
        for name in names:
            with cache.read(name,consume=False) as pred:raw=evaluator.metrics(pred)
            row={'candidate':name,'prediction_sha256':generation[name]['prediction_sha256'],
                 'raw_metrics':raw,**evaluator.aggregate(raw,floor,ceiling)}
            outcomes.append(row);emit('candidate_scored',seed=seed,**row)
        report['panels'].append({'seed':seed,'cutoff':cutoff,'target':target,'floor':floor,'ceiling':ceiling,'results':outcomes})
        save(RUN/'report.partial.json',report)
        del future,evaluator
    summary=[]
    for name in names:
        rows=[next(r for r in p['results'] if r['candidate']==name) for p in report['panels']]
        valid=all(r['calibration_valid'] for r in rows)
        summary.append({'candidate':name,'scores':[r['local_score'] for r in rows],
                        'mean_score':float(np.mean([r['local_score'] for r in rows])) if valid else None,
                        'raw_metrics':[r['raw_metrics'] for r in rows],'skills':[r['skills'] for r in rows],
                        'all_calibrations_valid':valid,
                        'mean_skills':{m:float(np.mean([r['skills'][m] for r in rows])) for m in rows[0]['skills']} if valid else None})
    by={r['candidate']:r for r in summary}
    control_names=spec.get("control_candidates",names[:2])
    controls=[by[n] for n in control_names];passing=[]
    for name in [n for n in names if n not in control_names]:
        new=by[name]
        passed=new['all_calibrations_valid'] and all(c['all_calibrations_valid'] for c in controls) and all(
            new['scores'][i]>max(c['scores'][i] for c in controls) for i in range(3)) and all(
            new['mean_skills'][m]>=max(c['mean_skills'][m] for c in controls) for m in new['mean_skills'])
        if passed:passing.append(name)
    contrast=None
    contrast_names=spec.get('contrast_candidates',names[2:4])
    if all(by[n]['all_calibrations_valid'] for n in contrast_names):
        contrast={m:by[contrast_names[1]]['mean_skills'][m]-by[contrast_names[0]]['mean_skills'][m] for m in by[contrast_names[1]]['mean_skills']}
    report.update(status='completed',completed_utc=now(),summary=summary,
                  passing_candidates=passing, growth_enabled_minus_disabled_mean_skills=contrast,
                  expand_to_16_resamples=any(by[n]['mean_score']>=60 for n in passing),
                  scope=spec['scope'],plan_sha256=digest(RUN/'plan.json'),
                  resource_peak_process_working_set_bytes=peak_memory())
