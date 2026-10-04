"""Reusable cutoff-only full-gene context; same established source-domain preparation."""
from pathlib import Path
from collections import Counter
import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.decomposition import PCA
from scnode_past_fold_training import PastOnlyMatrix, past_features
from tigon_conditional_fullpanel import panel_values
from run_t1 import digest
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def prepare(spec,RUN,emit):
    cutoff=spec['cutoff'];prepared=HERE/'private/associated_prepared_01';panel_path=ROOT/'outputs/t1_run/T1__val.genes.txt'
    panel=panel_path.read_text().splitlines()
    if len(panel)!=32285 or len(set(panel))!=len(panel): raise ValueError('Original panel mismatch')
    raw_x=np.load(prepared/'expression.npy',mmap_mode='r')
    stages=pd.read_csv(prepared/'selected_metadata.csv').numeric_stage.to_numpy(float)
    symbols=pd.read_csv(prepared/'genes.csv').symbol.fillna('').tolist()
    x=PastOnlyMatrix(raw_x,stages,cutoff)
    past=np.flatnonzero(stages<=cutoff)
    np.testing.assert_array_equal(np.unique(stages[past]),spec['fit_stages'])
    anchor_rows=np.flatnonzero(stages==cutoff)
    counts=Counter(symbols);lookup={g:i for i,g in enumerate(symbols) if g and counts[g]==1}
    mapped=np.array([i for i,g in enumerate(panel) if g in lookup],dtype=int)
    columns=np.array([lookup[panel[i]] for i in mapped],dtype=int)
    if len(mapped)!=26775: raise ValueError('Unique mapped panel changed')
    protected=np.setdiff1d(np.arange(len(panel)),mapped)
    for label,values in [('fit_rows',past),('mapped',mapped),('columns',columns),('anchor_rows',anchor_rows)]:
        np.save(RUN/(label+'.npy'),values)
    donor_rows=np.random.default_rng(20260928).choice(anchor_rows,1500,replace=False)
    np.testing.assert_array_equal(donor_rows,np.random.default_rng(20260928).choice(anchor_rows,1500,replace=False))
    np.save(RUN/'donor_rows.npy',donor_rows)
    donors=panel_values(x,donor_rows,mapped,columns,len(panel))
    anchor_path=RUN/f'source_E{cutoff}_anchor.h5ad'
    anchor=ad.AnnData(sparse.csr_matrix(panel_values(x,anchor_rows,mapped,columns,len(panel))),
                     obs=pd.DataFrame(index=[str(i) for i in anchor_rows]),
                     var=pd.DataFrame(index=panel))
    anchor.write_h5ad(anchor_path);del anchor
    emit('past_training_scope_frozen',fit_max_stage=float(stages[past].max()),fit_cells=len(past),
         anchor_cells=len(anchor_rows),mapped_genes=len(mapped),anchor_sha256=digest(anchor_path))
    atlas_features=past_features(x,past,columns,4096)
    atlas_guard=past_features(x,past,columns,384)
    official_lookup={g:i for i,g in enumerate(panel)}
    features=np.array([official_lookup[symbols[i]] for i in atlas_features])
    guard=np.array([official_lookup[symbols[i]] for i in atlas_guard])
    values=np.asarray(x[np.ix_(past,atlas_features)],np.float32)
    center=values.mean(0);scale=np.maximum(values.std(0),.1)
    normalized=(values-center)/scale
    fit=np.random.default_rng(20260928).choice(len(past),min(3000,len(past)),replace=False)
    pca=PCA(n_components=8,random_state=20260928).fit(normalized[fit])
    coordinates=pca.transform(normalized)
    whitening=np.maximum(coordinates.std(0),.1)
    basis=pca.components_/whitening[:,None];coordinates=coordinates/whitening
    np.savez_compressed(RUN/'fresh_encoder.npz',features=features,guard_features=guard,center=center,
                        scale=scale,basis=basis,pca_center=pca.mean_,whitening=whitening,pca_rows=past[fit])
    del values,normalized
    return {k:v for k,v in locals().items() if k in ("raw_x","stages","x","past","anchor_rows","mapped","columns","protected","donors","panel","symbols","atlas_features","features","guard","coordinates","basis","pca","center","scale","anchor_path")}
