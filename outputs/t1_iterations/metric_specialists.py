"""Select development specialists from complete, matched four-metric vectors."""
import json
from pathlib import Path
import numpy as np
from train_extended_atlas import HERE
from run_t1 import digest


RUNS = ['challenge_backtest_01','challenge_identity_01','detection_backtest_01',
    'hurdle_backtest_01','guard_backtest_01','neighborhood_backtest_01','window_backtest_01','state_search_01','specialist_integration_01','de_specialist_search_01','reliability_backtest_01','kernel_backtest_01','variogram_backtest_01']


def analyze():
    rows = []; sources = {}; reference = None
    for name in RUNS:
        path = HERE/'private'/name/'report.json'
        report = json.loads(path.read_text()); sources[name] = digest(path)
        signature = [(p['seed'],p['floor'],p['ceiling']) for p in report['panels']]
        if reference is None: reference = signature
        elif signature != reference: raise ValueError('Cannot compare unmatched calibration panels')
        for summary in report['summaries']:
            if not summary['all_calibrations_valid']: continue
            results = [next(r for r in p['results'] if r['candidate']==summary['candidate']) for p in report['panels']]
            skills = {k:float(np.mean([r['skills'][k] for r in results])) for k in results[0]['skills']}
            rows.append({'run':name,'candidate':summary['candidate'],'mean_score':summary['mean_score'],
                'minimum_score':summary['minimum_score'],'mean_skills':skills})
    specialists = {k:max(rows,key=lambda r:r['mean_skills'][k]) for k in rows[0]['mean_skills']}
    front = [r for r in rows if not any(all(s['mean_skills'][k]>=r['mean_skills'][k] for k in r['mean_skills'])
        and any(s['mean_skills'][k]>r['mean_skills'][k] for k in r['mean_skills']) for s in rows)]
    return {'specialists':specialists,'pareto_front':front,'candidates':rows,'source_report_sha256':sources,
        'selection_scope':'Reused E9.5 development evidence. Specialist best scores cannot be combined without scoring integrated predictions.',
        'official_72_verified':False}


if __name__ == '__main__':
    result = analyze()
    (HERE/'METRIC_SPECIALISTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'specialists':result['specialists'],'pareto_front':result['pareto_front']},indent=2))
