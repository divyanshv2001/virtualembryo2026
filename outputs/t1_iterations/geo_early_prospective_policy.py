"""Freeze independent QC/mapping rules and dry-run cached pilots; no acquisition."""
import gzip
import json
from collections import Counter, defaultdict

import pandas as pd
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event


def main():
    out = HERE/'private/geo_early_prospective_policy_01'
    if out.exists():
        raise ValueError('Preserve previous policy dry run')
    out.mkdir()
    annotation = HERE/'private/geo_early_biotype_audit_02/gene_annotation.json'
    atlas = HERE/'private/associated_prepared_01/genes.csv'
    panel_path = HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt'
    schema_path = HERE/'GSE76118_EARLY_SCHEMA_RESULTS.json'
    policy = {'created_utc': now(), 'code_sha256': digest(HERE/'geo_early_prospective_policy.py'),
        'inputs_sha256': {str(p.relative_to(HERE)): digest(p) for p in (annotation, atlas, schema_path)},
        'panel_sha256': digest(panel_path),
        'eligibility': 'Strict CD1 E8.5/E9.5 manifest only; mutants excluded, B6 controls separate. E9.5 never trains E8.5 validation.',
        'prospective_qc': {'minimum_assigned_gene_counts': 1_000_000,
            'minimum_protein_coding_assigned_fraction': .8,
            'maximum_unknown_biotype_assigned_fraction': .01,
            'nonnegative_integer_counts_required': True,
            'reject_archived_paired_reads_below_one_million_when_verified': True},
        'qc_interpretation': 'Independent conservative count-depth screen, not exact reproduction of paper total-read QC or analyzed-cell IDs. Cohort assay/embryo validation still required.',
        'mapping_rule': 'Build stable-ID candidates from associated atlas and Ensembl79 names restricted to panel; retain only IDs having exactly one candidate panel symbol and symbols having exactly one candidate ID. Reject all ambiguity; no fuzzy aliases.',
        'unmapped_gene_policy': 'Unknown coverage, never zero-fill for a forecast; external data only informs mapped genes with matched incumbent backoff elsewhere.',
        'acquisition_budget_next_batch': {'maximum_cells': 24, 'maximum_compressed_expression_bytes': 48_000_000,
            'selection': 'Stage/region-stratified lexical GSM order frozen before expression download; retain QC failures, no replacement based on performance.'},
        'promotion_gate': 'No external forecast promotion without past-only matched-control full-panel temporal test; original >=64 replicate mean/lower-tail >72 gate unchanged.',
        'models_fit': 0, 'full_panel_scores': 0}
    (out/'plan.json').write_text(json.dumps(policy, indent=2)+'\n')
    (out/'executed_source.py').write_bytes((HERE/'geo_early_prospective_policy.py').read_bytes())
    append_event(out/'events.jsonl', 'policy_frozen', policy_sha256=digest(out/'plan.json'))
    genes = json.loads(annotation.read_text())
    panel = set(panel_path.read_text().splitlines())
    candidates = defaultdict(set)
    for gid, row in genes.items():
        if row['symbol'] in panel:
            candidates[gid].add(row['symbol'])
    for row in pd.read_csv(atlas).fillna('').itertuples():
        if row.symbol in panel and row.ensembl:
            candidates[row.ensembl.split('.')[0]].add(row.symbol)
    # Count every candidate edge, so ambiguous IDs cannot silently leave an alias free.
    multiplicity = Counter(symbol for symbols in candidates.values() for symbol in symbols)
    mapping = {gid:next(iter(symbols)) for gid,symbols in candidates.items()
               if len(symbols)==1 and multiplicity[next(iter(symbols))]==1}
    mapping_path = out/'stable_id_panel_mapping.json'
    mapping_path.write_text(json.dumps(mapping, sort_keys=True)+'\n')
    outcomes = []
    for sample in json.loads(schema_path.read_text())['outcomes']:
        path = HERE/'private/geo_early_schema_audit_01'/(sample['gsm']+'.HTSeq.output.txt.gz')
        if digest(path) != sample['file_sha256']:
            raise ValueError('Pilot checksum changed')
        with gzip.open(path,'rt') as stream:
            counts = [(g,float(v)) for g,v in (line.rstrip().split('\t') for line in stream) if not g.startswith('__')]
        total = sum(v for _,v in counts)
        protein = sum(v for g,v in counts if genes.get(g,{}).get('biotype')=='protein_coding')/total
        unknown = sum(v for g,v in counts if not genes.get(g,{}).get('biotype'))/total
        reasons = []
        if total < 1_000_000: reasons.append('assigned_gene_counts_below_1000000')
        if protein < .8: reasons.append('protein_fraction_below_0.8')
        if unknown > .01: reasons.append('unknown_biotype_fraction_above_0.01')
        if not all(v>=0 and v.is_integer() for _,v in counts): reasons.append('invalid_count_values')
        if sample['gsm']=='GSM2033420': reasons.append('verified_archived_paired_read_estimate_82198_below_1000000')
        outcomes.append({'gsm':sample['gsm'], 'stage':sample['stage'], 'prospective_count_qc_pass':not reasons,
            'rejection_reasons':reasons, 'original_paper_qc_passed':None,
            'mapped_panel_genes':sum(g in mapping for g,_ in counts),
            'mapped_assigned_count_fraction':sum(v for g,v in counts if g in mapping)/total})
    report = {'updated_utc':now(), 'policy_sha256':digest(out/'plan.json'),
        'mapping_sha256':digest(mapping_path), 'unique_panel_mapping_genes':len(mapping),
        'panel_size':len(panel), 'mapping_coverage_fraction':len(mapping)/len(panel),
        'mapping_coverage_interpretation':'Annotation union includes IDs absent from the historical count matrix; actual pilot coverage is reported per cell and is lower.',
        'ambiguous_stable_ids':sum(len(v)>1 for v in candidates.values()),
        'outcomes':outcomes, 'models_fit':0, 'full_panel_scores':0, 'reward_delta':0,
        'training_ready':False, 'next_gate':'Freeze 24-cell stage/region-only selection plan, audit cohort/QC/assay compatibility on disk before any transfer model.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (HERE/'GSE76118_PROSPECTIVE_QC_POLICY.json').write_text(json.dumps(policy,indent=2)+'\n')
    (HERE/'GSE76118_PROSPECTIVE_POLICY_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    append_event(out/'events.jsonl', 'policy_dry_run_completed', report_sha256=digest(out/'report.json'))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
