"""Decode the author-linked UCSC count matrix into a bounded staged sample."""
import json
import argparse
import struct
import sys
import zlib
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from iterate import append_event, now
sys.path.insert(0,str(HERE.parents[1]/'outputs/t1_run'))
from run_t1 import digest
SOURCE = HERE/'private/extended_atlas'
OUT = HERE/'private/extended_prepared_02'


def decode_record(compressed, cells):
    data = zlib.decompress(compressed)
    length = struct.unpack('<H',data[:2])[0]
    description = data[2:2+length].decode('ascii')
    values = np.frombuffer(data[2+length:],dtype='<f4')
    if len(values)!=cells: raise ValueError('Cell count differs from metadata')
    if not np.isfinite(values).all() or (values<0).any(): raise ValueError('Invalid source expression')
    return description, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', default='extended_prepared_02')
    parser.add_argument('--domain', choices=['whole', 'cardiac', 'cardiac_associated'], default='whole')
    parser.add_argument('--cells-per-stage', type=int, default=1000)
    args = parser.parse_args()
    if not 100 <= args.cells_per_stage <= 10000: raise ValueError('Invalid sample size')
    if not args.round.replace('_', '').isalnum(): raise ValueError('Invalid round')
    OUT = HERE/'private'/args.round
    if OUT.exists(): raise SystemExit('Prepared dataset exists; preserve it.')
    manifest = json.loads((SOURCE/'manifest.json').read_text())
    for record in manifest['files']:
        if digest(SOURCE/record['file'])!=record['sha256']: raise ValueError('Source changed')
    OUT.mkdir(parents=True)
    events = OUT/'events.jsonl'
    stage_values = np.arange(7.5 if args.domain != 'whole' else 6.5,9.51,.25)
    plan = {'created_before_execution_utc':now(),'seed':20260928,'cells_per_stage':args.cells_per_stage,
        'allowed_stages':stage_values.tolist(),'ambiguous_stage_policy':'Exclude mixed/unknown staging before sample selection.',
        'sample_selection':'Uniform seeded sample within each declared stage; metadata labels and supplied embeddings are not model inputs.',
        'normalization':'Require count-valued source; log1p(10000 * counts / full indexed-gene library sum).',
        'development_training_max_stage':7.5,'development_target':8.5,'final_training_max_stage':8.5,'final_target':9.5,
        'source_manifest':manifest,'code_sha256':digest(Path(__file__)),'submissions_used':0}
    cardiac_labels = ['Anterior cardiopharyngeal progenitors', 'Cardiomyocytes FHF 1',
        'Cardiomyocytes FHF 2', 'Cardiomyocytes SHF 1', 'Cardiomyocytes SHF 2',
        'Cardiopharyngeal progenitors', 'Cardiopharyngeal progenitors FHF',
        'Cardiopharyngeal progenitors SHF', 'Endocardium', 'Epicardium']
    if args.domain == 'cardiac_associated':
        cardiac_labels += ['Branchial arch neural crest', 'Migratory neural crest',
            'Pharyngeal mesoderm', 'Pharyngeal endoderm', 'Foregut', 'Gut tube',
            'Lateral plate mesoderm', 'Mesenchyme', 'Embryo proper mesothelium',
            'Embryo proper endothelium', 'Venous endothelium', 'Surface ectoderm',
            'Non-neural ectoderm', 'Blood progenitors', 'Haematoendothelial progenitors',
            'Erythroid', 'EMP', 'Megakaryocyte progenitors']
    plan['domain'] = args.domain
    if args.domain != 'whole':
        plan.update(domain_labels=cardiac_labels,
            sample_selection='Uniform seeded sample up to the declared stage cap within the fixed annotation set. All smaller stage pools retained. Labels restrict the cohort only; not learner features.',
            annotation_caveat='Published annotations may have used a jointly analyzed atlas; this is cohort selection, not a fully independent dissection replica.',
            development_training_max_stage=8., development_target=9.)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    (OUT/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    append_event(events,'preparation_plan_frozen',sha256=digest(OUT/'plan.json'))
    metadata = pd.read_csv(SOURCE/'meta.tsv',sep='\t',low_memory=False)
    if metadata.cellId.duplicated().any(): raise ValueError('Duplicate cells')
    numerical = pd.to_numeric(metadata.stage.astype(str).str.replace('E','',regex=False),errors='coerce')
    known = numerical.notna().to_numpy()
    if not np.isin(numerical[known],np.arange(6.5,9.51,.25)).all(): raise ValueError('Unexpected/protected stage; do not fit')
    index = json.loads((SOURCE/'exprMatrix.json').read_text())
    genes = sorted([(k,int(v[0]),int(v[1])) for k,v in index.items() if '|' in k],key=lambda r:r[1])
    if set(k for k in index if '|' not in k)!={'_range'} or index['_range']!=[0,0]:
        raise ValueError('Unexpected index metadata')
    rng = np.random.default_rng(plan['seed']); rows = []
    for stage in stage_values:
        domain = metadata.celltype_extended_atlas.isin(cardiac_labels).to_numpy() if args.domain != 'whole' else np.ones(len(metadata), dtype=bool)
        pool = np.flatnonzero((numerical.to_numpy()==stage) & domain)
        if not len(pool) or (args.domain == 'whole' and len(pool)<plan['cells_per_stage']): raise ValueError('Insufficient cells at declared stage')
        rows.extend(rng.choice(pool,min(len(pool),plan['cells_per_stage']),replace=False).tolist())
    rows = np.asarray(rows,dtype=np.int64)
    selected = metadata.iloc[rows].copy()
    selected['numeric_stage'] = numerical.iloc[rows].to_numpy()
    selected['source_row'] = rows
    selected.to_csv(OUT/'selected_metadata.csv',index=False)
    np.save(OUT/'source_rows.npy',rows,allow_pickle=False)
    gene_table = pd.DataFrame([(k.split('|',1)[0],k.split('|',1)[1]) for k,_,_ in genes],columns=['ensembl','symbol'])
    gene_table.to_csv(OUT/'genes.csv',index=False)
    if gene_table.ensembl.duplicated().any(): raise ValueError('Duplicate gene IDs')
    matrix = open_memmap(OUT/'expression.npy',mode='w+',dtype=np.float32,shape=(len(rows),len(genes)))
    libraries = np.zeros(len(rows),dtype=np.float64)
    with (SOURCE/'exprMatrix.bin').open('rb') as f:
        for g,(key,offset,length) in enumerate(genes):
            f.seek(offset); _,vector = decode_record(f.read(length),len(metadata))
            values = vector[rows]
            if np.any(values!=np.floor(values)): raise ValueError('Source is not raw integer-valued counts; normalization assumption fails')
            matrix[:,g] = values; libraries += values
            if (g+1)%4000==0: append_event(events,'genes_decoded',genes=g+1,total=len(genes))
    if (libraries<=0).any(): raise ValueError('Empty source library')
    for start in range(0,len(rows),256):
        end = min(start+256,len(rows))
        matrix[start:end] = np.log1p(matrix[start:end]*(10000/libraries[start:end,None]))
    matrix.flush(); del matrix
    np.save(OUT/'raw_library_sizes.npy',libraries)
    report = {'shape':[len(rows),len(genes)],'stages':stage_values.tolist(),
        'domain':args.domain, 'domain_labels':cardiac_labels if args.domain != 'whole' else None,
        'selected_stage_counts':selected.numeric_stage.value_counts().sort_index().to_dict(),
        'excluded_ambiguous_stage_cells':int((~known).sum()),'source_stage_counts':metadata.stage.value_counts().to_dict(),
        'raw_library_median':float(np.median(libraries)),'normalization_target':10000,
        'expression_sha256':digest(OUT/'expression.npy'),'metadata_sha256':digest(OUT/'selected_metadata.csv'),
        'genes_sha256':digest(OUT/'genes.csv'),'source_manifest_sha256':digest(SOURCE/'manifest.json'),
        'provided_embeddings_used':False,'embryo_id_status':'sample is an assay/capture identifier; embryo_version labels Original/Extended, not independent embryos.',
        'scope':'Staged atlas sample for local training; not a valid challenge submission panel.'}
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    append_event(events,'prepared',**report)


if __name__=='__main__': main()
