"""Full-panel E8.5-to-E9.5 challenge development test with no future-stage fitting."""
import argparse
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from detection_transfer import DetectionTransfer
from importance_source import IndexedRows,select_source
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
    parser.add_argument('--round', default='importance_backtest_01')
    parser.add_argument('--dataset', default='balanced_proxy_prepared_01')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--alignment', choices=['moments', 'identity'], default='identity')
    args = parser.parse_args()
    if not all(v.replace('_', '').isalnum() for v in [args.round, args.dataset]): raise ValueError('Invalid run name')
    root = HERE.parents[1]; data = HERE/'private'/args.dataset; out = HERE/'private'/args.round
    if out.exists() and not args.resume: raise ValueError('Preserve previous run history; use --resume')
    if args.resume and not out.exists(): raise ValueError('No search to resume')
    prepared = json.loads((data/'report.json').read_text())
    for name, key in [('expression.npy', 'expression_sha256'), ('selected_metadata.csv', 'metadata_sha256'), ('genes.csv', 'genes_sha256')]:
        if digest(data/name) != prepared[key]: raise ValueError('Prepared data changed')
    panel = (root/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
    spec = json.loads((root/'outputs/t1_run/index.json').read_text())['T1:val']
    anchor_path = root/'data/E8.5_RNA.h5ad'; target_path = root/'data/E9.5_RNA.h5ad'
    inputs = {p.name:digest(p) for p in [anchor_path, target_path]}
    incumbent=HERE/'private/state_search_01'
    incumbent_audit=json.loads((incumbent/'generation.json').read_text())['unit_k16']
    if digest(incumbent/'unit_k16.npy')!=incumbent_audit['prediction_sha256']:raise ValueError('Changed incumbent')
    configs=[{'name':'copy_last','method':'copy'},{'name':'previous_incumbent','method':'old'}]
    configs += [{'name':f'weighted_m{mix}_k{k}','method':'weighted','states':k,'scaling':'unit','mix':mix}
        for mix in [.25,.5,1.] for k in [8,16]]
    seeds = [20260928, 20260929, 20260930]
    plan = {'created_before_training_utc':now(), 'configs':configs, 'evaluation_seeds':seeds,
        'alignment':args.alignment,
        'hypothesis':'Broad source groups alone failed. Past-anchor state-density ratios may repair remaining source/anchor mismatch. Coarse teacher unit16; ratios clipped .125..8, blend .25/.5/1, unique half-source subsample, states8/16. No duplicate cells or future-stage calibration. Six frozen variants plus persistence and old incumbent.',
        'incumbent_sha256':digest(incumbent/'unit_k16.npy'),
        'atlas_fit_max_stage':8.5, 'challenge_anchor_stage':8.5, 'development_target_stage':9.5,
        'input_sha256':inputs, 'prepared_report_sha256':digest(data/'report.json'),
        'source_sha256':{p:digest(HERE/p) for p in ['importance_backtest.py', 'importance_source.py', 'detection_transfer.py', 'challenge_transfer.py', 'robust_population.py', 'constrained_forecast.py', 'transfer_genes.py']},
        'fit_scope':'Atlas rows <=E8.5 and sampled challenge E8.5 donors only; target E9.5 never enters representation, adaptation, factors, state assignment or forecast generation.',
        'panel':'Complete official 32285-gene order; missing/ambiguous atlas genes retain challenge donor expression. Library mass conserved within mapped genes.',
        'history':'Development evaluation, not a fresh blind test. Challenge E9.5 was supplied and inspected historically, but is excluded from this learner.',
        'sampling':'1500 fixed E8.5 donors, three declared E9.5 panels of 1000 truth plus 1000 ceiling cells. Panels may overlap and are not independent embryo replicates.',
        'selection':'Eligible development candidate requires all valid panels, every panel >72 and every metric skill >=50. E10.5 export additionally requires matching earlier temporal promotion evidence.',
        'score_scope':'Local E9.5 calibration only; not hidden E10.5 official anchors or certification.',
        'submissions_allowed':0, 'jev_requests_allowed':0}
    out.mkdir(parents=True, exist_ok=True); events = out/'events.jsonl'
    if args.resume:
        saved = json.loads((out/'plan.json').read_text())
        if any(plan[k] != saved[k] for k in plan if k != 'created_before_training_utc'): raise ValueError('Frozen plan changed; cannot resume')
        plan = saved
    else: (out/'plan.json').write_text(json.dumps(plan, indent=2))
    for name in plan['source_sha256']: (out/name).write_bytes((HERE/name).read_bytes())
    append_event(events, 'challenge_backtest_plan_frozen', sha256=digest(out/'plan.json'))
    x = np.load(data/'expression.npy', mmap_mode='r')
    stages = pd.read_csv(data/'selected_metadata.csv').numeric_stage.to_numpy(float)
    atlas_symbols = pd.read_csv(data/'genes.csv').symbol.fillna('').tolist()
    donors, source_rows = read_cells(anchor_path, panel, 1500, 20260928)
    np.save(out/'anchor_rows.npy', source_rows)
    teacher=DetectionTransfer(x,stages,8.5,donors,panel,atlas_symbols,states=16,alignment='identity',covariance_limit=.4,feature_scaling='unit')
    selections={mix:select_source(x,stages,teacher,8.5,mix) for mix in [.25,.5,1.]}
    for mix,(rows,audit) in selections.items():
        np.save(out/f'source_rows_{mix}.npy',rows)
        (out/f'source_selection_{mix}.json').write_text(json.dumps(audit,indent=2))
    del teacher
    registry = out/'generation.json'
    audits = json.loads(registry.read_text()) if registry.exists() else {}
    predictions = {}
    def checkpoint(phase, completed, required):
        payload = {'phase':phase,'completed':completed,'required':required,'plan_sha256':digest(out/'plan.json'),
            'resume_command':f'outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/importance_backtest.py --round {args.round} --dataset {args.dataset} --alignment {args.alignment} --resume'}
        (out/'checkpoint.json').write_text(json.dumps(payload,indent=2))
    for c in configs:
        name = c['name']; path = out/f'{name}.npy'
        if name in audits:
            if digest(path) != audits[name]['prediction_sha256']: raise ValueError('Saved forecast changed')
            predictions[name] = np.load(path,mmap_mode='r')
            continue
        if c['method'] == 'copy':
            prediction = donors.copy(); indices = np.arange(len(donors)); audit = {'scope':'Measured persistence'}
            features = np.arange(min(384,len(panel)))
        elif c['method']=='old':
            prediction=np.load(incumbent/'unit_k16.npy').copy();indices=np.load(incumbent/'unit_k16_indices.npy')
            audit={'scope':'Verified prior unit_k16 control','source_prediction_sha256':digest(incumbent/'unit_k16.npy')}
            features=np.arange(min(384,len(panel)))
        else:
            selected,selection_audit=selections[c['mix']]
            model=DetectionTransfer(IndexedRows(x,selected),stages[selected],8.5,donors,panel,atlas_symbols,states=c['states'],
                alignment=args.alignment,covariance_limit=.4,feature_scaling=c['scaling'])
            model.save(out/f'model_{name}.npz')
            prediction,indices,audit=model.predict_detection(9.5,.5,1.,.02)
            audit['source_selection']=selection_audit;features=model.official_features
        np.save(path,prediction); np.save(out/f'{name}_indices.npy',indices)
        audits[name] = {**distribution_audit(prediction,donors[indices],features),**audit,
            'prediction_sha256':digest(path),'shape':list(prediction.shape)}
        (out/'generation.json').write_text(json.dumps(audits,indent=2))
        predictions[name] = np.load(path,mmap_mode='r')
        append_event(events,'importance_backtest_forecast_completed',candidate=name,completed=len(audits),required=len(configs))
        checkpoint('forecast_generation',len(audits),len(configs))
        del prediction
        if c['method'] == 'weighted': del model
    append_event(events, 'all_predictions_frozen_before_target_read', candidates=list(predictions))
    core, _ = load_core()
    report = {'plan':plan, 'audits':audits, 'panels':[], 'official_score':None, 'official_72_verified':False,
        'submissions_used':0, 'jev_requests_used':0}
    if args.resume and (out/'report.partial.json').exists(): report = json.loads((out/'report.partial.json').read_text())
    completed_seeds = {p['seed'] for p in report['panels']}
    for seed in seeds:
        if seed in completed_seeds: continue
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
        checkpoint('scoring',len(report['panels']),len(seeds))
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
    checkpoint('completed',len(seeds),len(seeds))
    append_event(events, 'challenge_backtest_completed', best_candidate=best['candidate'] if best else None,
        best_mean_local_score=best['mean_score'] if best else None, development_eligible=report['development_eligible'])


if __name__ == '__main__':
    with threadpool_limits(limits=2): main()
