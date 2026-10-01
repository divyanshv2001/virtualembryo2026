"""Read-only production export audit. Never opens future expression or fits a model."""
import hashlib
import json
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = HERE / 'private/production_coverage_audit_01'
PUBLIC = HERE / 'PRODUCTION_COVERAGE_AUDIT_RESULTS.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def event(kind, **values):
    with (RUN / 'events.jsonl').open('a') as f:
        f.write(json.dumps({'utc': now(), 'event': kind, **values}) + '\n')


def dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def audit():
    spec = HERE / 'NEXT_PRODUCTION_COVERAGE_AUDIT.json'
    export = ROOT / 'outputs/t1_submissions/formula_progress_20260930_01'
    private = HERE / 'private/formula_progress_20260930_01'
    prepared = HERE / 'private/associated_prepared_01'
    artifact = export / 'T1_val__formula_progress_20260930_01.h5ad'
    anchor_path = ROOT / 'data/E9.5_RNA.h5ad'
    panel_path = ROOT / 'outputs/t1_run/T1__val.genes.txt'
    policy = json.loads(spec.read_text())
    plan = json.loads((export / 'plan.json').read_text())
    prior = json.loads((export / 'report.json').read_text())
    prep = json.loads((prepared / 'report.json').read_text())
    panel = panel_path.read_text().splitlines()
    executed = private / 'export_formula_progress_submission_run.py'
    paths = {'artifact': artifact, 'anchor': anchor_path, 'panel': panel_path,
             'export_plan': export / 'plan.json', 'export_report': export / 'report.json',
             'source_genes': prepared / 'genes.csv', 'source_metadata': prepared / 'selected_metadata.csv',
             'source_expression': prepared / 'expression.npy', 'executed_source': executed,
             'observed_rows': private / 'observed_rows.npy', 'donor_indices': private / 'donor_indices.npy'}
    provenance = {k: digest(v) for k, v in paths.items()}
    for name, expected in [('artifact', policy['artifact_sha256']),
                           ('anchor', plan['input_sha256']['anchor']),
                           ('panel', plan['input_sha256']['panel']),
                           ('export_plan', prior['plan_sha256']),
                           ('source_genes', prep['genes_sha256']),
                           ('source_metadata', prep['metadata_sha256']),
                           ('source_expression', prep['expression_sha256']),
                           ('executed_source', prior['export_source_at_run_sha256'])]:
        if provenance[name] != expected:
            raise ValueError('Provenance mismatch: ' + name)
    if (plan['observed_cutoff'], plan['target'], plan['source_max_stage']) != (9.5, 10.5, 9.5):
        raise ValueError('Unexpected production time plan')
    source_text = executed.read_text()
    if "program.predict(10.5, 'joint', 1., sampling='systematic')" not in source_text:
        raise ValueError('Executed forecast target call differs; inspect code')
    stages = pd.read_csv(paths['source_metadata']).numeric_stage.to_numpy(float)
    if not np.isfinite(stages).all() or stages.max() > 9.5:
        raise ValueError('Invalid or future source stages')
    symbols = pd.read_csv(paths['source_genes']).symbol.fillna('').tolist()
    counts = Counter(symbols)
    lookup = {s: i for i, s in enumerate(symbols) if s and counts[s] == 1}
    mapping = np.array([lookup.get(g, -1) for g in panel])
    strata = np.array(['unique_mapped' if g in lookup else 'ambiguous' if counts[g] > 1 else 'absent' for g in panel])
    x = np.load(paths['source_expression'], mmap_mode='r')
    source_rows = np.flatnonzero(stages == 9.5)
    if x.shape != (len(stages), len(symbols)) or not len(source_rows):
        raise ValueError('Invalid source dimensions / missing cutoff stage')
    rows = np.load(paths['observed_rows'])
    indices = np.load(paths['donor_indices'])
    if indices.shape != (1500,) or not np.issubdtype(indices.dtype, np.integer) or indices.min() < 0 or indices.max() >= len(rows):
        raise ValueError('Invalid donor indices')
    anchor = ad.read_h5ad(anchor_path, backed='r')
    pred = ad.read_h5ad(artifact, backed='r')
    try:
        if not anchor.var_names.is_unique or anchor.var_names.tolist() != panel or pred.var_names.tolist() != panel:
            raise ValueError('Gene order or uniqueness mismatch')
        expected = np.sort(np.random.default_rng(plan['donor_seed']).choice(anchor.n_obs, plan['donors'], replace=False))
        np.testing.assert_array_equal(rows, expected)
        np.testing.assert_array_equal(rows, np.load(HERE / 'private/anchorslope_progress_20260929_01/observed_rows.npy'))
        if pred.shape != (1500, len(panel)):
            raise ValueError('Artifact shape mismatch')
        n = len(panel)
        arrays = {k: np.zeros(n) for k in ['anchor_mean', 'forecast_mean', 'anchor_detection', 'forecast_detection',
                                          'anchor_positive_mean', 'forecast_positive_mean', 'changed_fraction',
                                          'new_detection_fraction', 'lost_detection_fraction', 'source_cutoff_mean', 'source_cutoff_detection']}
        masses = {k: np.zeros(1500) for k in ['anchor_all', 'forecast_all', 'anchor_mapped', 'forecast_mapped']}
        for start in range(0, n, 256):
            stop = min(start + 256, n)
            a = dense(anchor.X[rows, start:stop]).astype(np.float64)[indices]
            b = dense(pred.X[:, start:stop]).astype(np.float64)
            if not np.isfinite(a).all() or not np.isfinite(b).all() or (a < 0).any() or (b < 0).any():
                raise ValueError('Invalid expression block')
            sl = slice(start, stop)
            for prefix, values in [('anchor', a), ('forecast', b)]:
                detect = values > 0
                arrays[prefix + '_mean'][sl] = values.mean(0)
                arrays[prefix + '_detection'][sl] = detect.mean(0)
                arrays[prefix + '_positive_mean'][sl] = values.sum(0) / np.maximum(detect.sum(0), 1)
                abundance = np.expm1(values)
                masses[prefix + '_all'] += abundance.sum(1)
                masses[prefix + '_mapped'] += abundance[:, mapping[sl] >= 0].sum(1)
            arrays['changed_fraction'][sl] = (a != b).mean(0)
            arrays['new_detection_fraction'][sl] = ((a == 0) & (b > 0)).mean(0)
            arrays['lost_detection_fraction'][sl] = ((a > 0) & (b == 0)).mean(0)
            mapped = np.flatnonzero(mapping[sl] >= 0)
            if len(mapped):
                src = np.asarray(x[np.ix_(source_rows, mapping[sl][mapped])], dtype=np.float64)
                arrays['source_cutoff_mean'][start + mapped] = src.mean(0)
                arrays['source_cutoff_detection'][start + mapped] = (src > 0).mean(0)
            if start % 4096 == 0:
                event('chunk_audited', genes_through=stop)
        summaries = {}
        for name in ['all', 'unique_mapped', 'ambiguous', 'absent']:
            mask = np.ones(n, dtype=bool) if name == 'all' else strata == name
            active = arrays['anchor_detection'][mask] > 0
            changed = arrays['changed_fraction'][mask] > 0
            summaries[name] = {'genes': int(mask.sum()), 'anchor_active_genes': int(active.sum()),
                               'changed_genes': int(changed.sum()), 'unchanged_active_genes': int((active & ~changed).sum()),
                               'mean_log1p_sum_before': float(arrays['anchor_mean'][mask].sum()),
                               'mean_log1p_sum_after': float(arrays['forecast_mean'][mask].sum()),
                               'detected_entries_before': int(round(arrays['anchor_detection'][mask].sum() * 1500)),
                               'detected_entries_after': int(round(arrays['forecast_detection'][mask].sum() * 1500)),
                               'new_detection_entries': int(round(arrays['new_detection_fraction'][mask].sum() * 1500)),
                               'lost_detection_entries': int(round(arrays['lost_detection_fraction'][mask].sum() * 1500))}
        arrays_path = RUN / 'per_gene_audit.npz'
        np.savez_compressed(arrays_path, **arrays, source_mapping=mapping, strata=strata, **masses)
        rel = np.abs(masses['forecast_mapped'] - masses['anchor_mapped']) / np.maximum(masses['anchor_mapped'], 1e-12)
        mapped = mapping >= 0
        src_mismatch = {'mapped_mean_absolute_log1p_difference': float(np.mean(np.abs(arrays['source_cutoff_mean'][mapped] - arrays['anchor_mean'][mapped]))),
                        'mapped_mean_absolute_detection_difference': float(np.mean(np.abs(arrays['source_cutoff_detection'][mapped] - arrays['anchor_detection'][mapped])))}
        return {'status': 'completed', 'completed_utc': now(), 'plan_sha256': digest(spec), 'input_sha256': provenance,
                'time_policy': {'observed': 9.5, 'target': 10.5, 'horizon_days': 1., 'source_max': float(stages.max()),
                                'executed_target_call_verified': True},
                'donor_provenance_passed': True, 'gene_order_passed': True, 'strata': summaries,
                'source_cutoff_cells': int(len(source_rows)), 'source_anchor_comparison': src_mismatch,
                'unchanged_protected_genes_verified': bool((arrays['changed_fraction'][~mapped] == 0).all()),
                'mapped_mass_max_relative_error': float(rel.max()),
                'mean_unmapped_anchor_abundance_fraction': float(np.mean(1 - masses['anchor_mapped'] / np.maximum(masses['anchor_all'], 1e-12))),
                'reported_head_support_counts_only': {'source': prior['forecast_audit']['supported_genes'], 'anchor': prior['forecast_audit']['anchor_supported_genes']},
                'private_per_gene_sha256': digest(arrays_path), 'raw_metrics': None, 'skills': None, 'reward': 0,
                'scope': 'Read-only same-donor artifact audit; no future truth/model fitting/score. Abundance is expm1(log-normalized expression), not raw UMI depth. Source comparison is sampled source versus sampled challenge, not biological matched cells. Coverage limits are measured; causal effect on official52.16 is unproven.'}
    finally:
        anchor.file.close()
        pred.file.close()


def main():
    if RUN.exists() or PUBLIC.exists():
        raise ValueError('Do not duplicate production audit')
    RUN.mkdir(parents=True)
    event('audit_started', script_sha256=digest(Path(__file__)))
    try:
        report = audit()
    except Exception as exc:
        (RUN / 'traceback.txt').write_text(traceback.format_exc())
        report = {'status': 'failed', 'failed_utc': now(), 'error': type(exc).__name__ + ': ' + str(exc),
                  'raw_metrics': None, 'skills': None, 'reward': 0, 'scope': 'Read-only audit failure; no score.'}
    save(RUN / 'report.json', report)
    report['report_sha256'] = digest(RUN / 'report.json')
    save(PUBLIC, report)
    event('audit_finished', status=report['status'], report_sha256=report['report_sha256'])
    print(json.dumps({'status': report['status'], 'report': str(PUBLIC)}))


if __name__ == '__main__':
    main()
