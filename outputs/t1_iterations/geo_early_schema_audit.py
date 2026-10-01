"""Four bounded, genotype/stage-verified individual count files; no fitting."""
import gzip
import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event

MAX_BYTES=2_000_000
MAX_UNPACKED=8_000_000


def main():
    manifest_path=HERE/'GSE76118_WT_STAGE_MANIFEST.json'
    metadata_path=HERE/'private/geo_early_qc_audit_01/allowed_stage_metadata_corrected.json'
    manifest=json.loads(manifest_path.read_text())
    metadata={r['id']:r for r in json.loads(metadata_path.read_text())}
    selected=[]
    for stage in ('E8.5','E9.5'):
        candidates=[metadata[gsm] for gsm in manifest['allowed_sample_ids'][stage]]
        # Two distinct regions, each chosen lexicographically without expression access.
        regions=sorted({r['chars'][2][1] for r in candidates})
        for region in regions[:2]:
            record=min((r for r in candidates if r['chars'][2][1]==region),key=lambda r:r['id'])
            assert record['chars'][0][1].lower()=='cd1'
            assert record['chars'][1][1].lower()==stage.lower()
            assert not any(term in (record['title']+' '+record['source']).lower() for term in ('mut','knock','cre;','-/-'))
            assert len(record['supplementary'])==1
            address=urlparse(record['supplementary'][0])
            if address.hostname!='ftp.ncbi.nlm.nih.gov' or f'/{record["id"]}/suppl/' not in address.path:
                raise ValueError('Unexpected individual-GSM file path')
            if not address.path.endswith('.HTSeq.output.txt.gz'):
                raise ValueError('Unexpected file format')
            selected.append({'gsm':record['id'],'stage':stage,'region':region,
                             'url':'https://ftp.ncbi.nlm.nih.gov'+address.path})
    if len(selected)!=4:raise ValueError('Require four distinct verified early-stage files')
    out=HERE/'private/geo_early_schema_audit_01'
    if out.exists():raise ValueError('Retain prior audit; no overwrite/repeated downloads')
    plan={'created_before_acquisition_utc':now(),'code_sha256':digest(HERE/'geo_early_schema_audit.py'),
          'strict_manifest_sha256':digest(manifest_path),'metadata_sha256':digest(metadata_path),
          'samples':selected,'maximum_compressed_bytes_per_file':MAX_BYTES,
          'maximum_unpacked_bytes_per_file':MAX_UNPACKED,'models_fit':0,
          'scope':'File schema/count quality only, four cells are not representative or validated training data. E9.5 never trains an E8.5 backtest. No mixed-stage download.',
          'retention':'Small individual files retained privately on D for repeatable QC; no data committed.'}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (out/'executed_source.py').write_bytes((HERE/'geo_early_schema_audit.py').read_bytes())
    events=out/'events.jsonl';append_event(events,'plan_frozen',plan_sha256=digest(out/'plan.json'))
    outcomes=[]
    for sample in selected:
        row=dict(sample)
        try:
            request=urllib.request.Request(sample['url'],headers={'User-Agent':'virtualembryo-early-count-qc/1.0'})
            with urllib.request.urlopen(request,timeout=30) as response:
                payload=response.read(MAX_BYTES+1)
            if len(payload)>MAX_BYTES:raise ValueError('File exceeds bounded download')
            path=out/(sample['gsm']+'.HTSeq.output.txt.gz');path.write_bytes(payload)
            with gzip.open(path,'rb') as f:content=f.read(MAX_UNPACKED+1)
            if len(content)>MAX_UNPACKED:raise ValueError('Expanded file too large')
            parsed=[];special={};columns=Counter()
            for line in content.decode().splitlines():
                values=line.split('\t');columns[len(values)]+=1
                if len(values)!=2:raise ValueError('Expected identifier/count pairs')
                value=float(values[1])
                if not value>=0 or not value<float('inf'):raise ValueError('Invalid count')
                if values[0].startswith('__'):special[values[0]]=value
                else:parsed.append((values[0],value))
            ids=[gene for gene,_ in parsed]
            if len(set(ids))!=len(ids):raise ValueError('Duplicate gene IDs')
            row.update(status='parsed',compressed_bytes=len(payload),unpacked_bytes=len(content),
                file_sha256=digest(path),gene_rows=len(parsed),identifier_prefix_counts=dict(Counter(g[:7] for g in ids)),
                all_counts_integer=all(v.is_integer() for _,v in parsed),detected_gene_count=sum(v>0 for _,v in parsed),
                assigned_gene_count_sum=sum(v for _,v in parsed),special_counters=special,
                column_counts=dict(columns),total_sequencing_reads_available=False,
                protein_coding_fraction_available=False,library_unit='HTSeq assigned gene counts; not TPM/CPM',
                original_paper_qc_passed=None,
                original_paper_qc_reason='Total reads and gene biotypes/assignment denominator not yet established; cannot reproduce original QC from gene sum alone.')
        except (ValueError,OSError,UnicodeError) as exc:
            row.update(status='failed',error_type=type(exc).__name__,error=str(exc)[:250])
        outcomes.append(row);append_event(events,'individual_schema_audited',**row)
        (out/'report.partial.json').write_text(json.dumps({'status':'in_progress','outcomes':outcomes},indent=2)+'\n')
    report={'status':'completed','plan_sha256':digest(out/'plan.json'),'outcomes':outcomes,
            'models_fit':0,'full_panel_scores':0,'reward_delta':0,'official_submissions':0,
            'training_ready':False,'scope':plan['scope'],
            'next_gate':'Recover per-cell QC provenance or reproducible prospective QC with appropriate gene biotypes, then assess gene overlap and assay normalization on a larger strictly eligible early cohort before forecasting.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    public=dict(report,updated_utc=now(),report_sha256=digest(out/'report.json'))
    (HERE/'GSE76118_EARLY_SCHEMA_RESULTS.json').write_text(json.dumps(public,indent=2)+'\n')
    append_event(events,'schema_audit_completed',report_sha256=public['report_sha256'],training_ready=False)
    print(json.dumps(public))


if __name__=='__main__':main()
