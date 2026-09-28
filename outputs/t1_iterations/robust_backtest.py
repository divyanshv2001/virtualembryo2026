"""Repeat development folds with larger cohorts and guarded coarse states."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from constrained_forecast import PopulationForecast, distribution_audit
from robust_population import RobustPopulation
from train_extended_atlas import HERE
from offline_backtest import load_core, Panel
from iterate import now, append_event
from run_t1 import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--round', required=True)
    args = parser.parse_args()
    if not all(v.replace('_', '').isalnum() for v in [args.dataset, args.round]): raise ValueError('Invalid name')
    data = HERE/'private'/args.dataset; out = HERE/'private'/args.round
    if out.exists(): raise ValueError('Preserve run history')
    prepared = json.loads((data/'report.json').read_text())
    for name, key in [('expression.npy', 'expression_sha256'), ('selected_metadata.csv', 'metadata_sha256')]:
        if digest(data/name) != prepared[key]: raise ValueError('Input changed')
    configs = [{'name':'copy_last', 'states':4, 'tilt':0., 'expression':0.},
        {'name':'incumbent_global', 'states':4, 'tilt':.5, 'expression':.1}]
    configs += [{'name':f'robust_k{k}_t{t}_e{e}', 'states':k, 'tilt':t, 'expression':e}
        for k in [4, 8] for t, e in [(.5, 0.), (.5, .25), (.5, .5)]]
    folds = [(8., 8.25), (8., 9.), (8.25, 8.5), (8.5, 8.75), (8.5, 9.5), (8.75, 9.)]
    plan = {'created_before_training_utc':now(), 'configs':configs, 'folds':folds,
        'dataset':args.dataset, 'data_report_sha256':digest(data/'report.json'),
        'source_hashes':{p:digest(HERE/p) for p in ['robust_backtest.py', 'robust_population.py', 'constrained_forecast.py']},
        'history':'Repeated development stages; no fresh blind test or hidden E10.5 data.',
        'training':'All sampled past cells; 384 variance-selected genes, PCA16, KMeans4/8 with 5 initializations; three recent past-stage means. Reversing gene trends suppressed; uncertainty and small-state support shrinkage.',
        'constraints':'ESS >=80% donors; covariance Frobenius change <=10% over all 384 training features versus unchanged donors. Backoff uses donors only. Expression support retained and library totals 10000.',
        'sampling':'Fixed seed 20260928; up to 1000 donor cells and up to 1000 target cells per fold. Future target cells split into random halves for sampling calibration only.',
        'promotion':'Valid calibration on all folds; total >50 and each metric >=50 on every fold; mean >72. Failing candidates cannot transfer or submit.',
        'submissions_allowed':0, 'jev_requests_allowed':0}
    out.mkdir(parents=True); events = out/'events.jsonl'
    (out/'plan.json').write_text(json.dumps(plan, indent=2))
    for name in plan['source_hashes']: (out/name).write_bytes((HERE/name).read_bytes())
    append_event(events, 'robust_plan_frozen', sha256=digest(out/'plan.json'))
    x = np.load(data/'expression.npy', mmap_mode='r')
    metadata = pd.read_csv(data/'selected_metadata.csv')
    stages = metadata.numeric_stage.to_numpy(float)
    core, _ = load_core()
    report = {'plan':plan, 'folds':[], 'official_score':None, 'official_72_verified':False,
        'submissions_used':0, 'jev_requests_used':0}
    for cutoff in sorted(set(a for a, _ in folds)):
        models = {k:RobustPopulation(x, stages, cutoff, states=k) for k in [4, 8]}
        reference = models[4].donor
        for k, model in models.items():
            np.savez_compressed(out/f'model_{cutoff}_k{k}.npz', features=model.features, center=model.center,
                scale=model.scale, pca_components=model.pca.components_, pca_mean=model.pca.mean_,
                latent_slope=model.latent_slope, abundance_slope=model.abundance_slope, donor_latent=model.z,
                cluster_centers=model.clusterer.cluster_centers_, population_slope=model.population_slope,
                state_slope=model.state_slope, state_support=model.state_support, donor_labels=model.donor_labels,
                donor_selection=model.donor_selection, cutoff=np.array(cutoff))
        append_event(events, 'guarded_models_fitted', cutoff=cutoff, training_cells=int((stages <= cutoff).sum()),
            state_support={str(k):m.state_support.tolist() for k, m in models.items()})
        for a, target in folds:
            if a != cutoff: continue
            predictions = {}; audits = {}
            for c in configs:
                model = models[c['states']]
                if c['name'] == 'incumbent_global':
                    prediction, donor, indices, weights = PopulationForecast.predict(model, target, c['tilt'], c['expression'])
                    guard = {'guard_applied':False}
                else:
                    prediction, donor, indices, weights = model.predict(target, c['tilt'], c['expression'])
                    guard = {'guard_applied':True, **model.last_audit}
                predictions[c['name']] = prediction
                audits[c['name']] = {**distribution_audit(prediction, donor, model.features), **guard,
                    'effective_sample_size':float(1/np.sum(weights**2)), 'unique_donors':int(len(np.unique(indices)))}
                np.save(out/f'indices_{cutoff}_{target}_{c["name"]}.npy', indices)
            append_event(events, 'predictions_frozen_before_scoring', cutoff=cutoff, target=target, candidates=list(predictions))
            target_rows = np.flatnonzero(stages == target)
            rng = np.random.default_rng(20260928)
            selected = rng.choice(target_rows, min(1000, len(target_rows)), replace=False)
            truth_rows, ceiling_rows = np.array_split(selected, 2)
            panel = Panel(core, np.asarray(x[truth_rows]), reference, 20260928)
            floor = panel.metrics(reference); ceiling = panel.metrics(np.asarray(x[ceiling_rows]))
            results = []
            for name, prediction in predictions.items():
                raw = panel.metrics(prediction)
                result = {'candidate':name, 'raw_metrics':raw, **panel.aggregate(raw, floor, ceiling)}
                results.append(result)
                append_event(events, 'robust_candidate_evaluated', cutoff=cutoff, target=target,
                    candidate=name, local_score=result['local_score'], calibration_valid=result['calibration_valid'])
            report['folds'].append({'cutoff':cutoff, 'target':target, 'floor':floor, 'ceiling':ceiling,
                'results':results, 'audits':audits, 'truth_cells':len(truth_rows), 'reference_cells':len(reference)})
            (out/'report.partial.json').write_text(json.dumps(report, indent=2))
            del predictions, panel
        del models
    summaries = []
    for c in configs:
        rows = [next(r for r in f['results'] if r['candidate'] == c['name']) for f in report['folds']]
        valid = all(r['calibration_valid'] for r in rows); scores = [r['local_score'] for r in rows]
        summaries.append({'candidate':c['name'], 'scores':scores, 'calibration_valid':valid,
            'mean_score':float(np.mean(scores)) if valid else None, 'minimum_score':min(scores) if valid else None,
            'promotion_gate':bool(valid and min(scores)>50 and np.mean(scores)>72 and all(min(r['skills'].values()) >= .5 for r in rows))})
    report.update(status='completed', summaries=summaries, eligible_candidates=[r['candidate'] for r in summaries if r['promotion_gate']])
    (out/'report.json').write_text(json.dumps(report, indent=2))
    append_event(events, 'robust_backtest_completed', eligible_candidates=report['eligible_candidates'])


if __name__ == '__main__':
    with threadpool_limits(limits=2): main()
