"""Project all permitted source cells into the archived 8D encoder by gene streaming.

Only the 8D coordinates are retained; no full cell-by-feature matrix is made.
"""
from collections import Counter
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from train_extended_atlas import HERE
from prepare_extended_atlas import decode_record
from run_t1 import digest
from iterate import now,append_event


def main():
    root=HERE.parents[1];source=HERE/'private/extended_atlas'
    stats=HERE/'private/full_atlas_stage_means_01'
    old=HERE/'private/cnf_feature_challenge_01'
    out=HERE/'private/full_source_latent_coordinates_01'
    if out.exists():raise ValueError('Preserve previous coordinate run')
    report=json.loads((stats/'report.json').read_text())
    if report.get('status')!='completed':raise ValueError('Require complete full-source library sizes')
    panel=(root/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
    encoder=np.load(old/'encoder4096.npz')
    features=encoder['features'];center=encoder['center'];scale=encoder['scale']
    pca_center=encoder['pca_center'];basis=encoder['basis']
    meta=pd.read_csv(source/'meta.tsv',sep='\t',usecols=['stage'],low_memory=False)
    stage=pd.to_numeric(meta.stage.astype(str).str.replace('E','',regex=False),errors='coerce').to_numpy(float)
    rows=np.flatnonzero(np.isin(stage,[7.5,7.75,8.,8.25,8.5]))
    library=np.load(stats/'source_library_sizes.npy',mmap_mode='r')
    if len(rows)!=len(library):raise ValueError('Library/source row mismatch')
    index=json.loads((source/'exprMatrix.json').read_text())
    all_genes=[(key,int(pos[0]),int(pos[1])) for key,pos in index.items() if '|' in key]
    counts=Counter(key.split('|',1)[1] for key,_,_ in all_genes)
    by_symbol={key.split('|',1)[1]:(offset,length) for key,offset,length in all_genes
               if counts[key.split('|',1)[1]]==1}
    records=[(j,*by_symbol[panel[f]]) for j,f in enumerate(features)]
    if len(records)!=4096:raise ValueError('Archived feature count changed')
    plan={'created_utc':now(),'source_rows':len(rows),'features':len(features),'coordinates':len(basis),
          'max_source_stage':8.5,'encoder_sha256':digest(old/'encoder4096.npz'),
          'library_sha256':digest(stats/'source_library_sizes.npy'),
          'source_manifest_sha256':digest(source/'manifest.json'),
          'input_sha256':{f:digest(source/f) for f in ['meta.tsv','exprMatrix.json','exprMatrix.bin']},
          'code_sha256':digest(HERE/'full_source_latent_coordinates.py'),
          'method':'For every archived encoder feature, stream raw full-source counts, normalize by complete per-cell library, and accumulate the frozen PCA projection in 64-feature blocks. No full cell-feature matrix retained.',
          'memory_budget':'One 116175x64 float32 block plus 116175x8 accumulators; <512 MB algorithmic working arrays.',
          'scope':'Past-source coordinate preparation only; no future expression, target fit or forecast.',
          'submissions_allowed':0,'jev_requests_allowed':0}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2))
    events=out/'events.jsonl';append_event(events,'plan_frozen',sha256=digest(out/'plan.json'))
    z=np.zeros((len(rows),len(basis)),dtype=np.float64)
    with threadpool_limits(limits=2):
        with (source/'exprMatrix.bin').open('rb') as handle:
            for start in range(0,len(records),64):
                batch=records[start:start+64]
                block=np.empty((len(rows),len(batch)),dtype=np.float32)
                for k,(_,offset,length) in enumerate(batch):
                    handle.seek(offset)
                    _,vector=decode_record(handle.read(length),len(meta))
                    block[:,k]=np.log1p(vector[rows].astype(np.float64)*(10000/library)).astype(np.float32)
                sl=slice(start,start+len(batch))
                standardized=(block-center[sl])/scale[sl]-pca_center[sl]
                z+=standardized.astype(np.float64)@basis[:,sl].T
                if (start+len(batch))%512==0:
                    append_event(events,'features_projected',features=start+len(batch),total=len(records))
    if not np.isfinite(z).all():raise ValueError('Nonfinite projected coordinates')
    # The sampled source rows must reproduce the archived encoder projection.
    prepared=HERE/'private/associated_prepared_01'
    source_rows=np.load(prepared/'source_rows.npy')
    sample_meta=pd.read_csv(prepared/'selected_metadata.csv',usecols=['numeric_stage'])
    selected=np.flatnonzero(sample_meta.numeric_stage.to_numpy(float)<=8.5)
    locations=np.searchsorted(rows,source_rows[selected])
    if (rows[locations]!=source_rows[selected]).any():raise ValueError('Sample source rows not present')
    sampled=np.load(prepared/'expression.npy',mmap_mode='r')
    symbols=pd.read_csv(prepared/'genes.csv').symbol.fillna('').tolist()
    symbol_counts=Counter(symbols);lookup={s:i for i,s in enumerate(symbols) if s and symbol_counts[s]==1}
    atlas=np.array([lookup[panel[f]] for f in features])
    check=selected[::max(1,len(selected)//256)][:256]
    sampled_projection=(((np.asarray(sampled[np.ix_(check,atlas)],dtype=np.float64)-center)/scale)-pca_center)@basis.T
    local=np.searchsorted(rows,source_rows[check])
    max_error=float(np.max(abs(z[local]-sampled_projection)))
    if max_error>1e-3:raise ValueError(f'Full-source projection fails sampled overlap replay: {max_error}')
    np.savez_compressed(out/'coordinates.npz',z=z.astype(np.float32),stages=stage[rows].astype(np.float32),
                        source_rows=rows)
    complete={'status':'completed','plan_sha256':digest(out/'plan.json'),
              'coordinates_sha256':digest(out/'coordinates.npz'),
              'source_rows':len(rows),'sample_replay_rows':len(check),
              'sample_replay_max_abs_error':max_error,
              'estimated_working_array_bytes':int(len(rows)*(64+len(basis))*8),
              'full_feature_matrix_retained':False,'future_expression_used':False}
    (out/'report.json').write_text(json.dumps(complete,indent=2))
    append_event(events,'coordinates_completed',**complete)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        folder=HERE/'private/full_source_latent_coordinates_01'
        if folder.exists():
            (folder/'failure.json').write_text(json.dumps({'status':'execution_failed','type':type(exc).__name__,'error':str(exc)},indent=2))
            append_event(folder/'events.jsonl','execution_failed',type=type(exc).__name__,error=str(exc))
        raise
