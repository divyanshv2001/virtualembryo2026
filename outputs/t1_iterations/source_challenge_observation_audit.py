"""Past-only E8.5 source/challenge observation-scale and support audit."""
from collections import Counter
import json
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event


def collect(matrix, rows, genes, labels, block=128):
    n_gene = len(genes)
    gene_sum = np.zeros(n_gene, dtype=np.float64)
    positive_count = np.zeros(n_gene, dtype=np.int64)
    fg_sum = np.zeros(n_gene, dtype=np.float64)
    fg_count = np.zeros(n_gene, dtype=np.int64)
    detected = np.empty(len(rows), dtype=np.int32)
    abundance = np.empty(len(rows), dtype=np.float64)
    fg_detected, fg_abundance = [], []
    for start in range(0, len(rows), block):
        stop = min(start + block, len(rows))
        selected = rows[start:stop]
        raw = matrix[selected, :][:, genes]
        values = raw.toarray() if sparse.issparse(raw) else np.asarray(raw)
        values = values.astype(np.float32, copy=False)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('Nonfinite or negative expression')
        positive = values > 0
        gene_sum += values.sum(0, dtype=np.float64)
        positive_count += positive.sum(0)
        detected[start:stop] = positive.sum(1)
        abundance[start:stop] = np.expm1(values).sum(1, dtype=np.float64)
        fg = labels[start:stop] == 'Foregut'
        if fg.any():
            fg_sum += values[fg].sum(0, dtype=np.float64)
            fg_count += positive[fg].sum(0)
            fg_detected.extend(detected[start:stop][fg].tolist())
            fg_abundance.extend(abundance[start:stop][fg].tolist())
    return {'gene_sum': gene_sum, 'positive_count': positive_count,
            'foregut_sum': fg_sum, 'foregut_positive_count': fg_count,
            'detected': detected, 'abundance_proxy': abundance,
            'foregut_detected': np.asarray(fg_detected),
            'foregut_abundance_proxy': np.asarray(fg_abundance),
            'foregut_cells': len(fg_detected)}


def quantiles(x):
    return [float(v) for v in np.quantile(x, [.1, .25, .5, .75, .9])]


def main():
    root = HERE.parents[1]
    source = HERE / 'private/associated_prepared_01'
    out = HERE / 'private/source_challenge_observation_audit_01'
    if out.exists():
        raise ValueError('Preserve frozen observation audit')
    anchor_path = root / 'data/E8.5_RNA.h5ad'
    panel_path = root / 'outputs/t1_run/T1__val.genes.txt'
    panel = panel_path.read_text().splitlines()
    symbols = pd.read_csv(source / 'genes.csv').symbol.fillna('').tolist()
    counts = Counter(symbols)
    lookup = {s: i for i, s in enumerate(symbols) if s and counts[s] == 1}
    official = np.array([i for i, s in enumerate(panel) if s in lookup])
    atlas = np.array([lookup[panel[i]] for i in official])
    plan = {
        'created_utc': now(), 'code_sha256': digest(HERE / 'source_challenge_observation_audit.py'),
        'prepared_report_sha256': digest(source / 'report.json'),
        'challenge_sha256': digest(anchor_path), 'panel_sha256': digest(panel_path),
        'source_stage': 8.5, 'challenge_stage': 8.5, 'mapped_unique_genes': len(official),
        'read_batch_cells': 128,
        'statistics': 'Per-cell detected mapped genes and sum(expm1(log1p-normalized mapped expression)); per-gene prevalence and mean log1p; same-name Foregut subset as a conservative check.',
        'decision_rule': 'Flag an observation-scale mismatch if full-cohort median detected count or abundance proxy ratio challenge/source lies outside [0.8,1.25]. If both lie inside but gene-prevalence Spearman is below .75, flag heterogeneous expression/composition. These are descriptive screens, not proof of causal technical or biological origin.',
        'scope': 'Permitted E8.5 only. No E9.5 target, model fitting, forecast, scorer or official submission. Retain aggregates only.',
    }
    out.mkdir(); (out / 'plan.json').write_text(json.dumps(plan, indent=2))
    events = out / 'events.jsonl'; append_event(events, 'plan_frozen', sha256=digest(out / 'plan.json'))
    meta = pd.read_csv(source / 'selected_metadata.csv', low_memory=False)
    source_rows = np.flatnonzero(meta.numeric_stage.to_numpy(float) == 8.5)
    source_labels = meta.celltype_extended_atlas.fillna('unannotated').to_numpy(str)[source_rows]
    x = np.load(source / 'expression.npy', mmap_mode='r')
    with threadpool_limits(limits=2):
        sampled = collect(x, source_rows, atlas, source_labels)
        a = ad.read_h5ad(anchor_path, backed='r')
        try:
            if not a.var_names.is_unique or a.var_names.tolist() != panel:
                raise ValueError('Challenge panel mismatch')
            challenge_rows = np.arange(a.n_obs)
            challenge_labels = a.obs.celltype.fillna('unannotated').to_numpy(str)
            challenge = collect(a.X, challenge_rows, official, challenge_labels)
        finally:
            a.file.close()
    prevalence_source = sampled['positive_count'] / len(source_rows)
    prevalence_challenge = challenge['positive_count'] / len(challenge_rows)
    expression_source = sampled['gene_sum'] / len(source_rows)
    expression_challenge = challenge['gene_sum'] / len(challenge_rows)
    prevalence_rho = float(spearmanr(prevalence_source, prevalence_challenge).statistic)
    expression_rho = float(spearmanr(expression_source, expression_challenge).statistic)
    ratio_detected = float(np.median(challenge['detected']) / np.median(sampled['detected']))
    ratio_abundance = float(np.median(challenge['abundance_proxy']) / np.median(sampled['abundance_proxy']))
    flag_scale = bool(not (.8 <= ratio_detected <= 1.25 and .8 <= ratio_abundance <= 1.25))
    fg = {'source_cells': sampled['foregut_cells'], 'challenge_cells': challenge['foregut_cells']}
    if min(fg.values()) >= 30:
        fg.update(source_detected_quantiles=quantiles(sampled['foregut_detected']),
                  challenge_detected_quantiles=quantiles(challenge['foregut_detected']),
                  source_abundance_quantiles=quantiles(sampled['foregut_abundance_proxy']),
                  challenge_abundance_quantiles=quantiles(challenge['foregut_abundance_proxy']),
                  prevalence_spearman=float(spearmanr(
                      sampled['foregut_positive_count']/fg['source_cells'],
                      challenge['foregut_positive_count']/fg['challenge_cells']).statistic))
    report = {'status': 'completed', 'plan_sha256': digest(out / 'plan.json'),
              'source_rows': len(source_rows), 'challenge_rows': len(challenge_rows),
              'mapped_genes': len(official),
              'source_detected_quantiles': quantiles(sampled['detected']),
              'challenge_detected_quantiles': quantiles(challenge['detected']),
              'source_abundance_proxy_quantiles': quantiles(sampled['abundance_proxy']),
              'challenge_abundance_proxy_quantiles': quantiles(challenge['abundance_proxy']),
              'median_detected_ratio': ratio_detected,
              'median_abundance_proxy_ratio': ratio_abundance,
              'gene_prevalence_spearman': prevalence_rho,
              'gene_mean_log1p_spearman': expression_rho,
              'same_name_foregut': fg, 'observation_scale_mismatch_flag': flag_scale,
              'expression_composition_heterogeneity_flag': bool(not flag_scale and prevalence_rho < .75),
              'limits': 'Source E8.5 is a sampled atlas cohort; sum(expm1) on mapped normalized genes is a proxy, not raw UMI library. Source and challenge type ontologies are not aligned. Technical versus biological cause cannot be inferred.'}
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    np.savez_compressed(out / 'gene_aggregates.npz', official=official,
                        source_prevalence=prevalence_source.astype(np.float32),
                        challenge_prevalence=prevalence_challenge.astype(np.float32),
                        source_mean_log1p=expression_source.astype(np.float32),
                        challenge_mean_log1p=expression_challenge.astype(np.float32))
    public = dict(report, created_utc=now(), report_sha256=digest(out / 'report.json'),
                  aggregates_sha256=digest(out / 'gene_aggregates.npz'))
    (HERE / 'SOURCE_CHALLENGE_OBSERVATION_AUDIT.json').write_text(json.dumps(public, indent=2))
    append_event(events, 'audit_completed', observation_scale_mismatch_flag=flag_scale,
                 prevalence_rho=prevalence_rho)


if __name__ == '__main__':
    main()
