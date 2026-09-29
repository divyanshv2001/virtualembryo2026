"""Read-only feasibility audit for E8.5 challenge-to-source state matching.

No target expression, forecast, or selection from E9.5 is involved.
"""
from collections import Counter
import json
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now


def main():
    root=HERE.parents[1]
    prepared=HERE/'private/associated_prepared_01'
    old=HERE/'private/cnf_feature_challenge_01'
    anchor=root/'data/E8.5_RNA.h5ad'
    panel=(root/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
    x=np.load(prepared/'expression.npy',mmap_mode='r')
    meta=pd.read_csv(prepared/'selected_metadata.csv')
    symbols=pd.read_csv(prepared/'genes.csv').symbol.fillna('').tolist()
    counts=Counter(symbols);lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
    saved=np.load(old/'encoder4096.npz')
    features=saved['features']
    atlas=np.array([lookup[panel[i]] for i in features],dtype=int)
    center=saved['center'];scale=saved['scale']
    basis=saved['basis'];pca_center=saved['pca_center']
    rows=np.flatnonzero(meta.numeric_stage.to_numpy(float)==8.5)
    source=np.asarray(x[np.ix_(rows,atlas)],dtype=np.float32)
    z_source=((source-center)/scale-pca_center)@basis.T
    labels=meta.celltype_extended_atlas.fillna('unannotated').to_numpy(str)[rows]
    a=ad.read_h5ad(anchor,backed='r')
    try:
        if not a.var_names.is_unique or a.var_names.tolist()!=panel:
            raise ValueError('Challenge panel mismatch')
        z_anchor=np.empty((a.n_obs,len(basis)),dtype=np.float32)
        types=a.obs.celltype.fillna('unannotated').to_numpy(str)
        for start in range(0,a.n_obs,256):
            end=min(start+256,a.n_obs)
            block=a.X[start:end,features]
            values=block.toarray() if sparse.issparse(block) else np.asarray(block)
            z_anchor[start:end]=((values.astype(np.float32)-center)/scale-pca_center)@basis.T
    finally:a.file.close()
    source_nn=NearestNeighbors(n_neighbors=16).fit(z_source)
    source_dist,_=source_nn.kneighbors(z_source)
    challenge_dist,neighbors=source_nn.kneighbors(z_anchor)
    typical=float(np.median(source_dist[:,1]))
    names,code=np.unique(labels,return_inverse=True)
    nearest=code[neighbors]
    majority=np.empty(len(z_anchor),dtype=int)
    support=np.empty(len(z_anchor),dtype=float)
    for i,codes in enumerate(nearest):
        counts=np.bincount(codes,minlength=len(names))
        majority[i]=counts.argmax();support[i]=counts.max()/len(codes)
    by_type=[]
    for name in np.unique(types):
        mask=types==name
        picked=np.bincount(majority[mask],minlength=len(names))
        top=np.argsort(-picked)[:3]
        by_type.append({'challenge_type':name,'cells':int(mask.sum()),
                        'median_distance_ratio':float(np.median(challenge_dist[mask,0])/max(typical,1e-8)),
                        'median_neighbor_label_support':float(np.median(support[mask])),
                        'top_source_types':[{'label':str(names[j]),'cells':int(picked[j])} for j in top]})
    report={'created_utc':now(),'scope':'Past-only E8.5 representation feasibility; no E9.5 target or model scoring.',
            'source_rows':len(rows),'anchor_rows':len(z_anchor),'neighbors':16,
            'source_self_median_first_neighbor_distance':typical,
            'challenge_median_first_neighbor_distance':float(np.median(challenge_dist[:,0])),
            'challenge_to_source_distance_ratio':float(np.median(challenge_dist[:,0])/max(typical,1e-8)),
            'anchor_fraction_majority_support_ge_half':float((support>=.5).mean()),
            'anchor_fraction_distance_ratio_le_two':float((challenge_dist[:,0]/max(typical,1e-8)<=2).mean()),
            'by_challenge_type':by_type,
            'encoder_sha256':digest(old/'encoder4096.npz'),
            'anchor_input_sha256':digest(anchor),
            'limitations':'Nearest-source labels are not validated ontology mappings; source is a sampled atlas cohort, not the full raw atlas. Distances assess feasibility only.'}
    (HERE/'ANCHOR_SOURCE_MAPPING_DIAGNOSTIC.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
