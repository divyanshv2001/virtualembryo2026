"""Frozen rolling development experiment; never accesses hidden challenge targets."""
import argparse
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from constrained_forecast import PopulationForecast, distribution_audit
from train_extended_atlas import DATA, HERE, evaluate_stage
from offline_backtest import load_core
from iterate import now, append_event
from run_t1 import digest


def mapping_audit(data=DATA):
    genes = pd.read_csv(data/'genes.csv').fillna('')
    panel = (HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
    counts = genes.symbol.value_counts()
    unique = set(counts[counts == 1].index)-{''}
    duplicated = sorted(set(counts[counts > 1].index)-{''})
    missing = sorted(set(panel)-set(genes.symbol))
    audit = {'unique_mappable_official_genes':sum(g in unique for g in panel),
        'missing_official_genes':len(missing), 'ambiguous_atlas_symbols':duplicated,
        'policy':'Transfer only unique exact symbols. Missing or ambiguous genes retain challenge donor expression; no zero filling or implicit aggregation.',
        'challenge_input_diagnostics':{}}
    root = HERE.parents[1]
    for stage in ['E8.5', 'E9.5']:
        a = ad.read_h5ad(root/'data'/f'{stage}_RNA.h5ad', backed='r')
        try:
            block = a.X[:256].toarray().astype(float)
            audit['challenge_input_diagnostics'][stage] = {
                'sample_scope':'First 256 input cells; diagnostic only, not a population estimate.',
                'zero_fraction':float((block == 0).mean()),
                'implied_library_quantiles':np.quantile(np.expm1(block).sum(1), [0, .5, 1]).tolist(),
                'panel_matches':a.var_names.tolist() == panel}
        finally:
            a.file.close()
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', default='rolling_constrained_01')
    parser.add_argument('--family', choices=['global', 'state'], default='global')
    parser.add_argument('--dataset', default='extended_prepared_02')
    args = parser.parse_args()
    if not args.round.replace('_', '').isalnum(): raise ValueError('Invalid round')
    if not args.dataset.replace('_', '').isalnum(): raise ValueError('Invalid dataset')
    data = HERE/'private'/args.dataset
    out = HERE/'private'/args.round
    if out.exists(): raise ValueError('Preserve previous run; choose a new round')
    prepared = json.loads((data/'report.json').read_text())
    for filename, key in [('expression.npy', 'expression_sha256'), ('selected_metadata.csv', 'metadata_sha256'), ('genes.csv', 'genes_sha256')]:
        if key not in prepared or digest(data/filename) != prepared[key]: raise ValueError('Prepared data changed or checksum missing')
    folds = [(7.5, 7.75), (7.5, 8.5), (8., 8.25), (8., 9.), (8.5, 9.5)]
    if prepared.get('domain') == 'cardiac':
        folds = [(8., 8.25), (8., 9.), (8.25, 8.5), (8.5, 9.5)]
    configs = [{'name':'copy_last', 'tilt':0., 'expression':0.}]
    configs += [{'name':f'tilt_{t}_expr_{e}', 'tilt':t, 'expression':e}
        for t, e in [(.25, 0.), (.5, 0.), (0., .1), (0., .25), (.25, .1), (.5, .1)]]
    model_class = PopulationForecast
    if args.family == 'state':
        from state_population import StatePopulationForecast
        model_class = StatePopulationForecast
        configs = [{'name':'copy_last', 'tilt':0., 'expression':0.}]
        configs += [{'name':f'state_tilt_{t}_expr_{e}', 'tilt':t, 'expression':e}
            for t, e in [(.5, 0.), (1., 0.), (0., .5), (0., 1.), (.5, .5), (1., .5)]]
    plan = {'created_before_execution_utc':now(), 'folds':folds, 'configs':configs,
        'code_sha256':digest(Path(__file__)), 'model_code_sha256':digest(HERE/'constrained_forecast.py'),
        'data_report_sha256':digest(data/'report.json'), 'dataset':args.dataset,
        'history':'All folds are development. E8.5 and E9.5 were read in earlier experiments. No fresh blind claim.',
        'selection':'Require valid calibration on every fold, total >50 on every fold, every metric skill >=50 on every fold, and mean total >72. Otherwise no transfer or submission.',
        'training':'Training-only 384-feature PCA, 16 dimensions; last three stage means estimate slopes. Covariance ridge=1, donor weights in [0.5,2] before normalization, abundance factors in [0.8,1.25].',
        'scope':'Atlas-local panels, no official score certification.', 'submissions_allowed':0, 'jev_requests_allowed':0}
    plan['family'] = args.family
    if args.family == 'state':
        plan['state_model_sha256'] = digest(HERE/'state_population.py')
        plan['training'] += ' State family: 24 past-only k-means states, 5 initializations; smoothed population log-frequency slopes; conditional abundance slopes require >=8 cells per state per recent stage and shrink by n/(n+32). Factors bounded to [2/3,1.5].'
    out.mkdir(parents=True)
    events = out/'events.jsonl'
    (out/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    for source in [Path(__file__), HERE/'constrained_forecast.py']:
        (out/source.name).write_bytes(source.read_bytes())
    if args.family == 'state':
        (out/'state_population.py').write_bytes((HERE/'state_population.py').read_bytes())
    append_event(events, 'rolling_plan_frozen', sha256=digest(out/'plan.json'))
    x = np.load(data/'expression.npy', mmap_mode='r')
    stages = pd.read_csv(data/'selected_metadata.csv').numeric_stage.to_numpy(float)
    core, _ = load_core()
    report = {'plan':plan, 'mapping_audit':mapping_audit(data), 'folds':[], 'official_score':None,
        'official_72_verified':False, 'submissions_used':0, 'jev_requests_used':0}
    for cutoff in sorted(set(a for a, _ in folds)):
        model = model_class(x, stages, cutoff)
        np.savez_compressed(out/f'model_{cutoff}.npz', features=model.features, center=model.center,
            scale=model.scale, pca_components=model.pca.components_, pca_mean=model.pca.mean_,
            latent_slope=model.latent_slope, abundance_slope=model.abundance_slope,
            donor_latent=model.z, cutoff=np.array(cutoff))
        if args.family == 'state':
            np.savez_compressed(out/f'states_{cutoff}.npz', centers=model.clusterer.cluster_centers_,
                donor_labels=model.donor_labels, population_slope=model.population_slope,
                state_slope=model.state_slope, state_support=model.state_support)
        append_event(events, 'past_only_model_fitted', cutoff=cutoff, training_cells=int((stages <= cutoff).sum()))
        for a, target in folds:
            if a != cutoff: continue
            predictions = {}; audits = {}
            for c in configs:
                prediction, donor, indices, weights = model.predict(target, c['tilt'], c['expression'])
                predictions[c['name']] = prediction
                audits[c['name']] = distribution_audit(prediction, donor, model.features[:128])
                audits[c['name']].update(unique_donors=int(len(np.unique(indices))),
                    effective_sample_size=float(1/np.sum(weights**2)))
                np.save(out/f'donors_{cutoff}_{target}_{c["name"]}.npy', indices)
            append_event(events, 'forecasts_frozen_before_target_scoring', cutoff=cutoff, target=target, candidates=list(predictions))
            metrics = evaluate_stage(core, x, stages, model.donor, predictions, target, events)
            metrics.update(training_cutoff=cutoff, distribution_audits=audits)
            report['folds'].append(metrics)
            (out/'report.partial.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            del predictions
        del model
    summaries = []
    for c in configs:
        rows = [next(r for r in f['results'] if r['candidate'] == c['name']) for f in report['folds']]
        valid = all(r['calibration_valid'] for r in rows)
        scores = [r['local_score'] for r in rows]
        summaries.append({'candidate':c['name'], 'scores':scores, 'all_calibrations_valid':valid,
            'mean_score':float(np.mean(scores)) if valid else None,
            'minimum_score':float(min(scores)) if valid else None,
            'promotion_gate':bool(valid and min(scores)>50 and np.mean(scores)>72
                and all(min(r['skills'].values()) >= .5 for r in rows))})
    report.update(status='completed', summaries=summaries,
        eligible_candidates=[s['candidate'] for s in summaries if s['promotion_gate']])
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    append_event(events, 'rolling_run_completed', eligible_candidates=report['eligible_candidates'])


if __name__ == '__main__':
    with threadpool_limits(limits=2): main()
