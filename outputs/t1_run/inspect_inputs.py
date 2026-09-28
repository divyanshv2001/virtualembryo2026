"""Read-only T1 metadata/expression integrity inspection in bounded blocks."""
import hashlib
import json
from pathlib import Path
import anndata as ad
import numpy as np
from scipy import sparse

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def inspect(path,panel):
    a=ad.read_h5ad(path,backed='r')
    try:
        names=a.var_names.tolist()
        if not a.var_names.is_unique:
            raise ValueError('Duplicate genes in '+path.name)
        lo,hi,zeros,values=0.0,0.0,0,0
        for start in range(0,a.n_obs,256):
            block=a.X[start:start+256]
            x=block.data if sparse.issparse(block) else np.asarray(block)
            if not np.isfinite(x).all() or (x<0).any():
                raise ValueError('Invalid expression in '+path.name)
            if x.size:
                lo=min(lo,float(x.min()));hi=max(hi,float(x.max()))
            values+=int(np.prod(block.shape))
            zeros+=int(np.prod(block.shape)-np.count_nonzero(x))
        return {'file':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path),
                'shape':[a.n_obs,a.n_vars],'X_storage':type(a.X).__name__,'genes_unique':True,
                'official_panel_exact_order':names==panel,'official_panel_same_set':set(names)==set(panel),
                'missing_official_genes':len(set(panel)-set(names)),'extra_genes':len(set(names)-set(panel)),
                'obs_columns':a.obs.columns.tolist(),'obsm_keys':list(a.obsm.keys()),'uns_keys':list(a.uns.keys()),
                'expression':{'finite':True,'nonnegative':True,'min':lo,'max':hi,'zero_fraction':zeros/values},
                'note':'Training metadata/integrity only. No future target or official predictive performance measured.'}
    finally:
        a.file.close()

if __name__=='__main__':
    panel=(HERE/'T1__val.genes.txt').read_text().splitlines()
    rows=[inspect(ROOT/'data'/name,panel) for name in ('E8.5_RNA.h5ad','E9.5_RNA.h5ad')]
    (HERE/'input_inspection.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))
