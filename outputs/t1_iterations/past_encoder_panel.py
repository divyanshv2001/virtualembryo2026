"""Past-only variable-gene PCA with complete selection provenance."""
from collections import Counter
import numpy as np
from sklearn.decomposition import PCA


def fit_past_encoder(x,stages,cutoff,symbols,panel,budget=4096,dimensions=8):
    past=np.flatnonzero(stages<=cutoff)
    if len(past)<=dimensions:raise ValueError('Insufficient past cells')
    counts=Counter(symbols);panel_set=set(panel);official={v:i for i,v in enumerate(panel)}
    allowed=np.array([i for i,v in enumerate(symbols) if v and v in panel_set and counts[v]==1])
    if budget>len(allowed) or dimensions>=budget:raise ValueError('Unsupported encoder budget')
    variance=np.zeros(x.shape[1])
    for start in range(0,x.shape[1],512):variance[start:start+512]=np.asarray(x[past,start:start+512],float).var(0)
    atlas=np.sort(allowed[np.argsort(-variance[allowed],kind='stable')[:budget]])
    features=np.array([official[symbols[i]] for i in atlas])
    raw=np.asarray(x[np.ix_(past,atlas)],np.float32);center=raw.mean(0);scale=np.maximum(raw.std(0),.1)
    normalized=(raw-center)/scale;fit=np.random.default_rng(20260928).choice(len(past),min(3000,len(past)),replace=False)
    pca=PCA(n_components=dimensions,random_state=20260928).fit(normalized[fit]);coordinates=pca.transform(normalized)
    whitening=np.maximum(coordinates.std(0),.1);basis=pca.components_/whitening[:,None]
    return {'past_rows':past,'pca_rows':past[fit],'features':features,'atlas_features':atlas,'center':center,'scale':scale,'basis':basis,'pca_center':pca.mean_,'whitening':whitening,'coordinates':coordinates/whitening}
