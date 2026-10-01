"""Frozen 24-cell early CD1 acquisition and QC; no forecasts or model fitting."""
import gzip
import json
import urllib.request
from collections import Counter
from urllib.parse import urlparse

import numpy as np
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event


def main():
    out = HERE/'private/geo_early_cohort_audit_01'
    if out.exists():
        raise ValueError('Existing cohort audit must be retained; no duplicate run')
    manifest_path = HERE/'GSE76118_WT_STAGE_MANIFEST.json'
    metadata_path = HERE/'private/geo_early_qc_audit_01/allowed_stage_metadata_corrected.json'
    policy_path = HERE/'GSE76118_PROSPECTIVE_QC_POLICY.json'
    manifest = json.loads(manifest_path.read_text())
    metadata = {r['id']:r for r in json.loads(metadata_path.read_text())}
    policy = json.loads(policy_path.read_text())
    selected = []
    for stage in ('E8.5','E9.5'):
        by_region = {}
        for gsm in sorted(manifest['allowed_sample_ids'][stage]):
            record = metadata[gsm]
            if record['chars'][0][1].lower()!='cd1' or record['chars'][1][1].lower()!=stage.lower():
                raise ValueError('Manifest/metadata stage or strain mismatch')
            by_region.setdefault(record['chars'][2][1], []).append(record)
        # Round-robin lexical region/ID selection, independent of counts and outcomes.
        chosen = []
        for index in range(12):
            for region in sorted(by_region):
                if index < len(by_region[region]):
                    chosen.append(by_region[region][index])
                    if len(chosen)==12: break
            if len(chosen)==12: break
        if len(chosen)!=12: raise ValueError('Insufficient strict stage candidates')
        for record in chosen:
            if len(record['supplementary'])!=1: raise ValueError('Unexpected file list')
            parsed = urlparse(record['supplementary'][0])
            if parsed.hostname!='ftp.ncbi.nlm.nih.gov' or f'/{record["id"]}/suppl/' not in parsed.path or not parsed.path.endswith('.HTSeq.output.txt.gz'):
                raise ValueError('Unexpected individual sample URL')
            selected.append({'gsm':record['id'], 'stage':stage, 'region':record['chars'][2][1],
                             'url':'https://ftp.ncbi.nlm.nih.gov'+parsed.path})
    out.mkdir()
    plan = {'created_before_acquisition_utc':now(), 'code_sha256':digest(HERE/'geo_early_cohort_audit.py'),
            'manifest_sha256':digest(manifest_path), 'metadata_sha256':digest(metadata_path),
            'policy_sha256':digest(policy_path), 'samples':selected,
            'max_compressed_bytes_per_file':2_000_000, 'max_total_compressed_bytes':48_000_000,
            'max_expanded_bytes_per_file':8_000_000, 'seed':None,
            'selection':'12 per stage, round-robin lexically sorted region then GSM; no outcome-based replacements',
            'scope':'Eligible early CD1 cohort/QC only; no mixed-stage resource, transfer fit, forecast or score. Retain failures.'}
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (out/'executed_source.py').write_bytes((HERE/'geo_early_cohort_audit.py').read_bytes())
    events = out/'events.jsonl'
    append_event(events,'plan_frozen',plan_sha256=digest(out/'plan.json'))
    annotation = json.loads((HERE/'private/geo_early_biotype_audit_02/gene_annotation.json').read_text())
    mapping = json.loads((HERE/'private/geo_early_prospective_policy_01/stable_id_panel_mapping.json').read_text())
    pilot = {r['gsm']:r for r in json.loads((HERE/'GSE76118_EARLY_SCHEMA_RESULTS.json').read_text())['outcomes']}
    rules = policy['prospective_qc']; rows=[]; total_bytes=0
    for sample in selected:
        row=dict(sample)
        try:
            cached = HERE/'private/geo_early_schema_audit_01'/(sample['gsm']+'.HTSeq.output.txt.gz')
            if sample['gsm'] in pilot and cached.exists():
                if digest(cached)!=pilot[sample['gsm']]['file_sha256']: raise ValueError('Cached pilot changed')
                path=cached; reused=True
            else:
                path=out/(sample['gsm']+'.HTSeq.output.txt.gz'); reused=False
                with urllib.request.urlopen(sample['url'],timeout=30) as response:
                    payload=response.read(2_000_001)
                if len(payload)>2_000_000: raise ValueError('Compressed file bound exceeded')
                path.write_bytes(payload)
            total_bytes+=path.stat().st_size
            if total_bytes>48_000_000: raise ValueError('Cohort compressed bound exceeded')
            with gzip.open(path,'rb') as stream: content=stream.read(8_000_001)
            if len(content)>8_000_000: raise ValueError('Expanded file bound exceeded')
            counts={};special={}
            for line in content.decode().splitlines():
                gid,number=line.split('\t'); value=float(number)
                if not np.isfinite(value) or value<0 or not value.is_integer(): raise ValueError('Invalid count')
                if gid.startswith('__'): special[gid]=value
                else:
                    if gid in counts: raise ValueError('Duplicate gene ID')
                    counts[gid]=value
            total=sum(counts.values())
            if total<=0: raise ValueError('Empty library')
            protein=sum(v for g,v in counts.items() if annotation.get(g,{}).get('biotype')=='protein_coding')/total
            unknown=sum(v for g,v in counts.items() if not annotation.get(g,{}).get('biotype'))/total
            reasons=[]
            if total<rules['minimum_assigned_gene_counts']: reasons.append('assigned_gene_counts_below_1000000')
            if protein<rules['minimum_protein_coding_assigned_fraction']: reasons.append('protein_fraction_below_0.8')
            if unknown>rules['maximum_unknown_biotype_assigned_fraction']: reasons.append('unknown_biotype_fraction_above_0.01')
            if sample['gsm']=='GSM2033420': reasons.append('verified_archived_read_estimate_below_1000000')
            transformed=np.log1p(10000*np.array([v for g,v in counts.items() if g in mapping])/total)
            row.update(status='parsed', file_sha256=digest(path), compressed_bytes=path.stat().st_size,
                reused_pilot=reused, gene_rows=len(counts), assigned_gene_count_sum=total,
                detected_genes=sum(v>0 for v in counts.values()), protein_fraction=protein,
                unknown_biotype_fraction=unknown, mapped_panel_genes=len(transformed),
                mapped_assigned_mass_fraction=sum(v for g,v in counts.items() if g in mapping)/total,
                log1p_10k_mapped_quantiles=dict(zip(('min','median','p99','max'),np.quantile(transformed,[0,.5,.99,1]).tolist())),
                prospective_qc_pass=not reasons, rejection_reasons=reasons, original_paper_qc_passed=None,
                embryo_replication_verified=False, library_unit='HTSeq assigned counts')
        except Exception as exc:
            row.update(status='failed',error_type=type(exc).__name__,error=str(exc)[:250],prospective_qc_pass=False)
        rows.append(row); append_event(events,'cohort_cell_audited',**row)
        (out/'report.partial.json').write_text(json.dumps({'status':'in_progress','outcomes':rows},indent=2)+'\n')
    report={'status':'completed','updated_utc':now(),'plan_sha256':digest(out/'plan.json'),
        'policy_sha256':digest(policy_path),'outcomes':rows,'compressed_bytes_including_reused':total_bytes,
        'parsed_cells':sum(r['status']=='parsed' for r in rows),
        'prospective_qc_passes_by_stage':dict(Counter(r['stage'] for r in rows if r.get('prospective_qc_pass'))),
        'failed_cells':sum(r['status']=='failed' for r in rows), 'models_fit':0,'full_panel_scores':0,
        'reward_delta':0,'training_ready':False,
        'limitations':'Small cardiac region cohort differs from whole embryo atlas and assays; original paper QC and embryo independence unresolved. No missing-gene zero imputation. Summary QC does not establish predictive transfer.',
        'next_gate':'Evaluate lineage/assay compatibility using available past-only atlas reference and eligible mapped genes; matched negative controls required before full-panel transfer forecast.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    public=dict(report,report_sha256=digest(out/'report.json'))
    (HERE/'GSE76118_EARLY_COHORT_RESULTS.json').write_text(json.dumps(public,indent=2)+'\n')
    append_event(events,'cohort_audit_completed',report_sha256=public['report_sha256'])
    print(json.dumps({k:v for k,v in public.items() if k!='outcomes'}))


if __name__=='__main__': main()
