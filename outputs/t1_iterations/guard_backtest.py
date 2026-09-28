"""Full-panel E8.5-to-E9.5 challenge development test with no future-stage fitting."""
import argparse
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from hurdle_transfer import HurdleTransfer
from constrained_forecast import distribution_audit
from train_extended_atlas import HERE
from offline_backtest import load_core, Panel
from iterate import append_event, now
from run_t1 import digest, validate


def read_cells(path, panel, count, seed):
    a = ad.read_h5ad(path, backed='r')
    try:
        if not a.var_names.is_unique or a.var_names.tolist() != panel: raise ValueError('Challenge panel mismatch')
        rows = np.sort(np.random.default_rng(seed).choice(a.n_obs, count, replace=False))
        block = a.X[rows, :]
        values = block.toarray() if sparse.issparse(block) else np.asarray(block)
        values = values.astype(np.float32)
        if not np.isfinite(values).all() or (values < 0).any(): raise ValueError('Invalid challenge values')
        return values, rows
    finally: a.file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', default='guard_backtest_01')
    parser.add_argument('--dataset', default='associated_prepared_01')
    parser.add_argument('--alignment', choices=['moments', 'identity'], default='identity')
    args = parser.parse_args()
    if not all(v.replace('_', '').isalnum() for v in [args.round, args.dataset]): raise ValueError('Invalid run name')
    root = HERE.parents[1]; data = HERE/'private'/args.dataset; out = HERE/'private'/args.round
    if out.exists(): raise ValueError('Preserve previous run history')
    prepared = json.loads((data/'report.json').read_text())
    for name, key in [('expression.npy', 'expression_sha256'), ('selected_metadata.csv', 'metadata_sha256'), ('genes.csv', 'genes_sha256')]:
        if digest(data/name) != prepared[key]: raise ValueError('Prepared data changed')
    panel = (root/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
    spec = json.loads((root/'outputs/t1_run/index.json').read_text())['T1:val']
    anchor_path = root/'data/E8.5_RNA.h5ad'; target_path = root/'data/E9.5_RNA.h5ad'
    inputs = {p.name:digest(p) for p in [anchor_path, target_path]}
    configs = [{'name':'copy_last', 'method':'copy', 'states':8, 'tilt':0., 'expression':0., 'detection':0., 'cap':.02, 'positive':False, 'covariance_limit':.1},
        {'name':'detection_incumbent', 'method':'state', 'states':8, 'tilt':.5, 'expression':.25, 'detection':.5, 'cap':.02, 'positive':False, 'covariance_limit':.1}]
    configs += [{'name':f'{kind}_e{e}_cov{cov}', 'method':'state', 'states':8,
        'tilt':.5, 'expression':e, 'detection':.5, 'cap':.02, 'positive':kind=='positive', 'covariance_limit':cov}
        for kind,e,cov in [('positive',.5,.2), ('positive',1.,.4), ('joint',.5,.2), ('joint',1.,.4)]]
    seeds = [20260928, 20260929, 20260930]
    plan = {'created_before_training_utc':now(), 'configs':configs, 'evaluation_seeds':seeds,
        'alignment':args.alignment,
        'hypothesis':'Guard ablation: stronger conditional or joint abundance changes may be suppressed by the donor-covariance constraint. Four declared settings compare 0.2/0.4 covariance limits against the unchanged 0.1 incumbent; evaluation anchors and scoring unchanged.',
        'atlas_fit_max_stage':8.5, 'challenge_anchor_stage':8.5, 'development_target_stage':9.5,
        'input_sha256':inputs, 'prepared_report_sha256':digest(data/'report.json'),
        'source_sha256':{p:digest(HERE/p) for p in ['guard_backtest.py', 'hurdle_transfer.py', 'detection_transfer.py', 'challenge_transfer.py', 'robust_population.py', 'constrained_forecast.py', 'transfer_genes.py']},
        'fit_scope':'Atlas rows <=E8.5 and sampled challenge E8.5 donors only; target E9.5 never enters representation, adaptation, factors, state assignment or forecast generation.',
        'panel':'Complete official 32285-gene order; missing/ambiguous atlas genes retain challenge donor expression. Library mass conserved within mapped genes.',
        'history':'Development evaluation, not a fresh blind test. Challenge E9.5 was supplied and inspected historically, but is excluded from this learner.',
        'sampling':'1500 fixed E8.5 donors, three declared E9.5 panels of 1000 truth plus 1000 ceiling cells. Panels may overlap and are not independent embryo replicates.',
        'selection':'Eligible development candidate requires all valid panels, every panel >72 and every metric skill >=50. E10.5 export additionally requires matching earlier temporal promotion evidence.',
        'score_scope':'Local E9.5 calibration only; not hidden E10.5 official anchors or certification.',
        'submissions_allowed':0, 'jev_requests_allowed':0}
    out.mkdir(parents=True); events = out/'events.jsonl'
    (out/'plan.json').write_text(json.dumps(plan, indent=2))
    for name in plan['source_sha256']: (out/name).write_bytes((HERE/name).read_bytes())
    append_event(events, 'challenge_backtest_plan_frozen', sha256=digest(out/'plan.json'))
    x = np.load(data/'expression.npy', mmap_mode='r')
    stages = pd.read_csv(data/'selected_metadata.csv').numeric_stage.to_numpy(float)
    atlas_symbols = pd.read_csv(data/'genes.csv').symbol.fillna('').tolist()
    donors, source_rows = read_cells(anchor_path, panel, 1500, 20260928)
    np.save(out/'anchor_rows.npy', source_rows)
    models = {k:HurdleTransfer(x, stages, 8.5, donors, panel, atlas_symbols, states=k, alignment=args.alignment) for k in [8]}
    append_event(events, 'atlas_and_anchor_fitted', training_atlas_cells=int((stages <= 8.5).sum()), anchor_cells=len(donors),
        features={str(k):len(m.model.features) for k, m in models.items()})
    for k, model in models.items(): model.save(out/f'model_k{k}.npz')
    predictions = {}; audits = {}
    for c in configs:
        model = models[c['states']]
        model.covariance_limit = c['covariance_limit']
        prediction, indices, audit = (model.predict(9.5) if c['method'] == 'copy' else model.predict_hurdle(9.5, c['detection'], c['expression'], c['cap'], c['positive']))
        predictions[c['name']] = prediction
        np.save(out/f'{c["name"]}.npy', prediction)
        np.save(out/f'{c["name"]}_indices.npy', indices)
        audits[c['name']] = {**distribution_audit(prediction, donors[indices], model.official_features), **audit,
            'prediction_sha256':digest(out/f'{c["name"]}.npy'), 'shape':list(prediction.shape)}
    append_event(events, 'all_predictions_frozen_before_target_read', candidates=list(predictions))
    core, _ = load_core()
    report = {'plan':plan, 'audits':audits, 'panels':[], 'official_score':None, 'official_72_verified':False,
        'submissions_used':0, 'jev_requests_used':0}
    for seed in seeds:
        target, target_rows = read_cells(target_path, panel, 2000, seed)
        order = np.random.default_rng(seed).permutation(len(target))
        truth, ceiling = target[order[:1000]], target[order[1000:]]
        np.save(out/f'target_rows_{seed}.npy', target_rows)
        evaluation = Panel(core, truth, donors, seed)
        floor = evaluation.metrics(donors); top = evaluation.metrics(ceiling)
        results = []
        for name, prediction in predictions.items():
            raw = evaluation.metrics(prediction)
            result = {'candidate':name, 'raw_metrics':raw, **evaluation.aggregate(raw, floor, top)}
            results.append(result)
            append_event(events, 'challenge_candidate_evaluated', seed=seed, candidate=name,
                local_score=result['local_score'], calibration_valid=result['calibration_valid'])
        report['panels'].append({'seed':seed, 'floor':floor, 'ceiling':top, 'results':results})
        (out/'report.partial.json').write_text(json.dumps(report, indent=2))
        del evaluation, target, truth, ceiling
    summaries = []
    for c in configs:
        rows = [next(r for r in p['results'] if r['candidate'] == c['name']) for p in report['panels']]
        valid = all(r['calibration_valid'] for r in rows); scores = [r['local_score'] for r in rows]
        summaries.append({'candidate':c['name'], 'scores':scores, 'all_calibrations_valid':valid,
            'mean_score':float(np.mean(scores)) if valid else None,
            'minimum_score':min(scores) if valid else None,
            'development_72_gate':bool(valid and min(scores)>72 and all(min(r['skills'].values()) >= .5 for r in rows))})
    valid_candidates = [r for r in summaries if r['all_calibrations_valid']]
    best = max(valid_candidates, key=lambda r:r['mean_score']) if valid_candidates else None
    # This artifact forecasts observed E9.5. Its filename prevents confusion with an E10.5 submission.
    validation = None
    if best:
        name = best['candidate']
        path = out/f'BACKTEST_ONLY_E9.5__{name}.h5ad'
        artifact = ad.AnnData(X=sparse.csr_matrix(predictions[name]),
            obs=pd.DataFrame(index=[f'backtest_{i:05}' for i in range(len(donors))]), var=pd.DataFrame(index=panel))
        artifact.uns['forecast_stage'] = 'E9.5'
        artifact.uns['purpose'] = 'Observed-stage local development backtest; not an E10.5 submission.'
        artifact.write_h5ad(path, compression='gzip')
        validation = validate(path, panel, spec)
    # Earlier rolling checks remain a required gate; no configuration passed them.
    temporal = json.loads((HERE/'private/robust_associated_01/report.json').read_text())
    report['prior_temporal_report_sha256'] = digest(HERE/'private/robust_associated_01/report.json')
    earlier_passes = [r['candidate'] for r in temporal['summaries'] if r['promotion_gate']]
    if earlier_passes: raise ValueError('Temporal promotion evidence changed; explicitly map configurations before refit')
    report.update(status='completed', summaries=summaries, best_development=best,
        format_validation=validation, development_eligible=[r['candidate'] for r in summaries if r['development_72_gate']],
        prior_temporal_eligible=earlier_passes, future_refit_performed=False, future_submission_exported=False,
        decision='No E10.5 refit/export without both challenge development and earlier temporal promotion evidence.')
    for path in [anchor_path, target_path]:
        if digest(path) != inputs[path.name]: raise ValueError('Challenge input changed during backtest')
    (out/'report.json').write_text(json.dumps(report, indent=2))
    append_event(events, 'challenge_backtest_completed', best_candidate=best['candidate'] if best else None,
        best_mean_local_score=best['mean_score'] if best else None, development_eligible=report['development_eligible'])


if __name__ == '__main__':
    with threadpool_limits(limits=2): main()
