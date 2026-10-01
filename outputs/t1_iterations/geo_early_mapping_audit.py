"""Map only previously stage-verified early pilot files; no network or model."""
import gzip
import json
from collections import Counter
import pandas as pd

from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event


def main():
    source=HERE/'private/geo_early_schema_audit_01'
    report_path=HERE/'GSE76118_EARLY_SCHEMA_RESULTS.json'
    report=json.loads(report_path.read_text())
    annotation_path=HERE/'private/associated_prepared_01/genes.csv'
    annotation=pd.read_csv(annotation_path).fillna('')
    symbol_counts=Counter(annotation.symbol);id_counts=Counter(annotation.ensembl)
    panel_path=HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt'
    panel=panel_path.read_text().splitlines();panelset=set(panel)
    lookup={r.ensembl:r.symbol for r in annotation.itertuples()
            if r.symbol in panelset and r.symbol and symbol_counts[r.symbol]==1 and id_counts[r.ensembl]==1}
    rows=[]
    for outcome in report['outcomes']:
        if outcome['status']!='parsed':continue
        path=source/(outcome['gsm']+'.HTSeq.output.txt.gz')
        if digest(path)!=outcome['file_sha256']:raise ValueError('Pilot count file changed')
        with gzip.open(path,'rt') as f:
            records=[(line.split('\t')[0],float(line.split('\t')[1])) for line in f if not line.startswith('__')]
        matched=[(lookup[g],v) for g,v in records if g in lookup]
        if len({g for g,v in matched})!=len(matched):raise ValueError('Ambiguous panel mapping')
        rows.append({'gsm':outcome['gsm'],'stage':outcome['stage'],'region':outcome['region'],
            'mapped_unique_panel_genes':len(matched),'panel_size':len(panel),
            'panel_annotation_coverage_fraction':len(matched)/len(panel),
            'positive_mapped_genes':sum(v>0 for g,v in matched),
            'mapped_assigned_count_fraction':sum(v for g,v in matched)/sum(v for g,v in records),
            'normalization_candidate':'log1p(10000*gene_count/all_assigned_gene_count_sum), after QC; not fitted or scored',
            'unknown_panel_genes_policy':'Absent/ambiguous annotation is missing coverage, not biological zero.'})
    public={'updated_utc':now(),'code_sha256':digest(HERE/'geo_early_mapping_audit.py'),
            'schema_report_sha256':digest(report_path),'annotation_sha256':digest(annotation_path),
            'panel_sha256':digest(panel_path),'rows':rows,'models_fit':0,'full_panel_scores':0,
            'scope':'Existing atlas IDs/symbols are not complete GRCm38.79 biotype annotation. Four cells cannot certify population or temporal transfer.',
            'training_ready':False}
    (HERE/'GSE76118_EARLY_MAPPING_RESULTS.json').write_text(json.dumps(public,indent=2)+'\n')
    append_event(source/'events.jsonl','mapping_audit_completed',public_report_sha256=digest(HERE/'GSE76118_EARLY_MAPPING_RESULTS.json'),models_fit=0)
    print(json.dumps({'mapped_panel_genes':[r['mapped_unique_panel_genes'] for r in rows],
                      'training_ready':False}))


if __name__=='__main__':main()
