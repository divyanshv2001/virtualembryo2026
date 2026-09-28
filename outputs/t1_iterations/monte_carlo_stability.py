"""Resumable Monte Carlo stability checks of frozen challenge forecasts; no seed search."""
import argparse
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from challenge_backtest import read_cells
from offline_backtest import load_core, Panel
from train_extended_atlas import HERE
from iterate import now, append_event
from run_t1 import digest


def summarize(rows, temporal_passed=False):
    valid = bool(rows) and all(r['calibration_valid'] for r in rows)
    result = {'replicates':len(rows), 'invalid_replicates':sum(not r['calibration_valid'] for r in rows),
        'all_calibrations_valid':valid, 'final_gate_passed':False}
    if not valid:
        return {**result, 'mean_score':None, 'lower_tail_score':None}
    scores = np.array([r['local_score'] for r in rows])
    delta = scores-np.array([r['persistence_score'] for r in rows])
    skills = {k:float(np.mean([r['skills'][k] for r in rows])) for k in rows[0]['skills']}
    result.update(mean_score=float(scores.mean()), lower_tail_score=float(np.quantile(scores, .025)),
        upper_tail_score=float(np.quantile(scores, .975)), minimum_score=float(scores.min()), maximum_score=float(scores.max()),
        mean_gain=float(delta.mean()), lower_tail_gain=float(np.quantile(delta, .025)), mean_skills=skills,
        final_gate_passed=bool(len(rows)>=64 and scores.mean()>72 and np.quantile(scores, .025)>72
            and np.quantile(delta, .025)>0 and min(skills.values())>=.5 and temporal_passed))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', default='monte_carlo_pilot_01')
    parser.add_argument('--replicates', type=int, default=16)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not args.round.replace('_', '').isalnum() or not 2<=args.replicates<=256: raise ValueError('Invalid configuration')
    source = HERE/'private/challenge_identity_01'; out = HERE/'private'/args.round
    root = HERE.parents[1]; target_path = root/'data/E9.5_RNA.h5ad'
    panel_path = root/'outputs/t1_run/T1__val.genes.txt'; panel = panel_path.read_text().splitlines()
    candidates = ['state_k8_e0.25', 'incumbent_global']
    frozen_files = ['copy_last.npy']+[c+'.npy' for c in candidates]
    plan = {'created_before_execution_utc':now(), 'replicates':args.replicates,
        'replicate_seeds':np.random.SeedSequence(2026092801).generate_state(args.replicates).astype(int).tolist(),
        'source_run':source.name, 'candidates':candidates, 'prediction_hashes':{f:digest(source/f) for f in frozen_files},
        'target_sha256':digest(target_path), 'panel_sha256':digest(panel_path), 'code_sha256':digest(Path(__file__)),
        'sampling':'1500 paired bootstrap positions for reference and frozen predictions; new 2000-cell target draw per replicate, split into 1000 truth/1000 ceiling.',
        'scope':'Conditional stability of frozen predictors on one development stage; no model training, independent embryo interval or hidden E10.5 score.',
        'gate':'Final mean and empirical 2.5th percentile >72, positive paired lower-tail gain, all calibration valid, >=64 replicates, mean metric skills >=50 and temporal promotion evidence.',
        'pilot':args.replicates<64, 'submissions_allowed':0, 'jev_requests_allowed':0}
    if out.exists():
        if not args.resume: raise ValueError('Run exists; use --resume or choose a new round')
        saved = json.loads((out/'plan.json').read_text())
        for k in plan:
            if k != 'created_before_execution_utc' and plan[k] != saved[k]: raise ValueError('Frozen plan/input changed; cannot resume')
        plan = saved
        report = json.loads((out/'report.partial.json').read_text())
    else:
        if args.resume: raise ValueError('No run to resume')
        out.mkdir(parents=True)
        (out/'plan.json').write_text(json.dumps(plan, indent=2))
        (out/'executed_source.py').write_bytes(Path(__file__).read_bytes())
        report = {'plan':plan, 'replicates':[], 'status':'running', 'official_72_verified':False}
        (out/'report.partial.json').write_text(json.dumps(report, indent=2))
    events = out/'events.jsonl'
    append_event(events, 'monte_carlo_started_or_resumed', completed=len(report['replicates']), required=args.replicates,
        plan_sha256=digest(out/'plan.json'))
    reference = np.load(source/'copy_last.npy', mmap_mode='r')
    predictions = {c:np.load(source/(c+'.npy'), mmap_mode='r') for c in candidates}
    core, _ = load_core()
    for replicate in range(len(report['replicates']), args.replicates):
        seed = plan['replicate_seeds'][replicate]; rng = np.random.default_rng(seed)
        positions = rng.integers(0, len(reference), len(reference))
        bootstrap_reference = np.asarray(reference[positions])
        target, target_rows = read_cells(target_path, panel, 2000, seed)
        permutation = rng.permutation(2000)
        truth, ceiling = target[permutation[:1000]], target[permutation[1000:]]
        np.savez_compressed(out/f'sampling_{replicate:03}.npz', reference_positions=positions,
            target_rows=target_rows, target_permutation=permutation)
        evaluator = Panel(core, truth, bootstrap_reference, seed)
        floor = evaluator.metrics(bootstrap_reference); top = evaluator.metrics(ceiling)
        persistence = evaluator.aggregate(floor, floor, top)
        results = []
        for c, frozen in predictions.items():
            raw = evaluator.metrics(np.asarray(frozen[positions]))
            results.append({'candidate':c, 'raw_metrics':raw, **evaluator.aggregate(raw, floor, top),
                'persistence_score':persistence['local_score']})
        report['replicates'].append({'replicate':replicate, 'seed':seed, 'floor':floor, 'ceiling':top,
            'persistence':persistence, 'results':results})
        (out/'report.partial.json').write_text(json.dumps(report, indent=2))
        (out/'checkpoint.json').write_text(json.dumps({'completed_replicates':len(report['replicates']),
            'required_replicates':args.replicates, 'plan_sha256':digest(out/'plan.json'),
            'resume_command':f'outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/monte_carlo_stability.py --round {args.round} --replicates {args.replicates} --resume'}, indent=2))
        append_event(events, 'monte_carlo_replicate_completed', replicate=replicate,
            scores={r['candidate']:r['local_score'] for r in results})
        del evaluator, target, truth, ceiling, bootstrap_reference
    if digest(target_path) != plan['target_sha256']: raise ValueError('Target input changed')
    temporal = json.loads((HERE/'private/robust_associated_01/report.json').read_text())
    temporal_passed = any(r['promotion_gate'] for r in temporal['summaries'])
    # Promotion requires explicit same-configuration temporal evidence; none exists currently.
    if temporal_passed: raise ValueError('Map temporal evidence to these configurations before promotion')
    report['summaries'] = {c:summarize([next(r for r in rep['results'] if r['candidate']==c)
        for rep in report['replicates']], temporal_passed=False) for c in candidates}
    report.update(status='completed', objective_status='threshold_not_met', submissions_used=0, jev_requests_used=0)
    (out/'report.json').write_text(json.dumps(report, indent=2))
    append_event(events, 'monte_carlo_completed', summaries=report['summaries'])


if __name__ == '__main__':
    with threadpool_limits(limits=2): main()
