"""Reproducible sampled T1 persistence and optional public mean-shift baseline.

No target values, API calls or automatic upload. Sparse source matrices are
read in blocks; predictions and donor IDs stay in ignored local files.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def values(x):
    return x.data if sparse.issparse(x) else np.asarray(x)

def validate(path,panel,spec):
    a=ad.read_h5ad(path,backed='r')
    try:
        if a.var_names.tolist()!=panel or not a.var_names.is_unique:
            raise ValueError('Official ordered panel mismatch')
        if not spec['min_cells']<=a.n_obs<=spec['max_cells']:
            raise ValueError('Cell count outside official limits')
        if a.n_vars!=spec['n_genes'] or a.X.shape!=(a.n_obs,a.n_vars):
            raise ValueError('Expression shape mismatch')
        if list(a.obsm.keys()):
            raise ValueError('T1 export must carry no coordinates')
        dtype=str(a.X.dtype)
        if dtype!='float32':
            raise ValueError('This exporter promises float32')
        for start in range(0,a.n_obs,256):
            x=values(a.X[start:start+256])
            if not np.isfinite(x).all() or (x<0).any():
                raise ValueError('Invalid finite/nonnegative output')
        return {'passed':True,'shape':[a.n_obs,a.n_vars],'dtype':dtype,'panel_exact_order':True,
                'finite_nonnegative':True,'coordinates_absent':True,'obs_columns':a.obs.columns.tolist(),
                'sha256':digest(path),'bytes':path.stat().st_size,
                'scope':'Local format validation only; not official predictive score or competition eligibility'}
    finally:
        a.file.close()

def means(a,columns):
    if 'celltype' not in a.obs:
        raise ValueError('Missing training celltype; no shift candidate may be built')
    labels=a.obs['celltype'].astype('string')
    if labels.isna().any() or (labels=='').any():
        raise ValueError('Missing cell types require a separately specified rule')
    labels=labels.to_numpy(dtype=str)
    types=sorted(set(labels))
    total=np.zeros((len(types),len(columns)),dtype=np.float64)
    counts=np.zeros(len(types),dtype=np.int64)
    lookup={name:i for i,name in enumerate(types)}
    ids=np.array([lookup[v] for v in labels])
    for start in range(0,a.n_obs,256):
        block=a.X[start:start+256][:,columns]
        for k in np.unique(ids[start:start+256]):
            mask=ids[start:start+256]==k
            total[k]+=np.asarray(block[mask].sum(axis=0)).ravel()
            counts[k]+=int(mask.sum())
    return labels,{name:total[i]/counts[i] for i,name in enumerate(types)}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cells',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=20260928)
    parser.add_argument('--shift-candidate',action='store_true')
    args=parser.parse_args()
    spec=json.loads((HERE/'index.json').read_text())['T1:val']
    panel=(HERE/spec['genes_file']).read_text().splitlines()
    panel_hash=hashlib.sha256('\n'.join(panel).encode()).hexdigest()
    if panel_hash[:len(spec['genes_sha256'])]!=spec['genes_sha256']:
        raise ValueError('Official index/panel hash mismatch')
    if not spec['min_cells']<=args.cells<=spec['max_cells']:
        raise ValueError('Requested sample count outside official limits')
    paths=[ROOT/'data'/n for n in ('E8.5_RNA.h5ad','E9.5_RNA.h5ad')]
    stages=[ad.read_h5ad(p,backed='r') for p in paths]
    try:
        columns=[]
        for a in stages:
            if not a.var_names.is_unique:
                raise ValueError('Duplicate source genes')
            idx=a.var_names.get_indexer(panel)
            if (idx<0).any():
                raise ValueError('Missing official genes in source')
            columns.append(idx)
        last=stages[1]
        if last.n_obs<args.cells:
            raise ValueError('Too few source rows for sampling without replacement')
        rng=np.random.Generator(np.random.PCG64(args.seed))
        rows=np.sort(rng.choice(last.n_obs,args.cells,replace=False))
        source=last.X[rows,:][:,columns[1]].astype(np.float32)
        x=source.tocsr() if sparse.issparse(source) else sparse.csr_matrix(source)
        if not np.isfinite(x.data).all() or (x.data<0).any():
            raise ValueError('Invalid source expression')
        out=ad.AnnData(X=x,obs=pd.DataFrame(index=[f'prediction_{i:05}' for i in range(args.cells)]),var=pd.DataFrame(index=panel))
        target=HERE/'T1_val__sampled_copy_last.h5ad'
        out.write_h5ad(target,compression='gzip')
        checks=validate(target,panel,spec)
        reloaded=ad.read_h5ad(target)
        diff=reloaded.X-x
        if diff.nnz and np.any(diff.data!=0):
            raise ValueError('Output does not equal selected source rows')
        checks['source_rows_identical_after_roundtrip']=True
        np.save(HERE/'donor_rows.npy',rows)
        records=[{'method':'sampled_copy_last','artifact':target.name,'validation':checks,
                  'interpretation':'Seeded E9.5 empirical sample; not exactly the official baked floor or a proven future-stage improvement'}]
        if args.shift_candidate:
            labels_a,mu_a=means(stages[0],columns[0])
            labels_b,mu_b=means(last,columns[1])
            shared=set(mu_a)&set(mu_b)
            raw=x.toarray()
            selected=labels_b[rows]
            negative=0
            for label in np.unique(selected):
                if label not in shared:
                    continue
                mask=selected==label
                block=raw[mask].astype(np.float64)+(mu_b[label]-mu_a[label])
                negative+=int((block<0).sum())
                raw[mask]=np.maximum(block,0).astype(np.float32)
            candidate=ad.AnnData(X=sparse.csr_matrix(raw),obs=out.obs.copy(),var=out.var.copy())
            path=HERE/'T1_val__pseudobulk_shift_exploratory.h5ad'
            candidate.write_h5ad(path,compression='gzip')
            records.append({'method':'pseudobulk_shift','artifact':path.name,'validation':validate(path,panel,spec),
                            'shared_exact_annotation_labels':len(shared),'endpoint_only_last_labels':len(set(mu_b)-shared),
                            'clipped_negative_entries':negative,'hyperparameters':{'shift_multiplier':1,'target_stage':'E10.5'},
                            'interpretation':'Public-style per-celltype mean-shift comparator, exact-string label matching and clipping; no re-normalization; no measured target performance or annotation harmonization claim'})
        report={'board':'T1:val','target_stage':'E10.5','training_stages':['E8.5','E9.5'],
                'seed':args.seed,'rng':'PCG64','cells':args.cells,'gene_panel_hash':panel_hash,
                'source_files':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)} for p in paths],
                'versions':{m:importlib.metadata.version(m) for m in ('anndata','numpy','scipy','h5py','pandas')},
                'artifacts':records,'official_score':None,'uploaded':False,'live_jev_requests':0,
                'eligibility':'Not certified by this local run; audit exposure history remains recorded.',
                'source_preservation':'Inputs opened read-only; output contains no copied source annotations/coordinates/uns.'}
        (HERE/'run_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({'board':report['board'],'artifacts':records,'official_score':None,'uploaded':False},indent=2))
    finally:
        for a in stages:
            a.file.close()

if __name__=='__main__':
    main()
