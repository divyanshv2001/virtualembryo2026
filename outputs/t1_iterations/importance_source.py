"""Past-anchor state-density weighting with unique source-cell subsampling."""
import numpy as np

class IndexedRows:
    def __init__(self,base,rows):
        self.base=base;self.rows=np.asarray(rows,dtype=int);self.shape=(len(self.rows),base.shape[1])
        if len(np.unique(self.rows))!=len(self.rows):raise ValueError('Duplicate source cells are not independent observations')
    def __getitem__(self,key):
        if isinstance(key,tuple):return self.base[self.rows[key[0]],key[1]]
        return self.base[self.rows[key]]

def select_source(x,stages,teacher,cutoff,mix,seed=2026092811):
    if not 0<=mix<=1:raise ValueError('Invalid weight blend')
    m=teacher.model;rows=np.flatnonzero(stages<=cutoff)
    values=np.asarray(x[np.ix_(rows,m.features)],dtype=float)
    z=m.pca.transform(np.clip((values-m.center)/m.scale,-10,10));labels=m.clusterer.predict(z)
    k=len(m.clusterer.cluster_centers_)
    source=np.bincount(labels[stages[rows]==cutoff],minlength=k)+.5
    anchor=np.bincount(teacher.labels[teacher.trusted],minlength=k)+.5
    source/=source.sum();anchor/=anchor.sum()
    ratio=np.clip(anchor/source,.125,8.);desired=(1-mix)+mix*ratio[labels]
    rng=np.random.default_rng(seed);selected=[];ess={}
    for stage in np.unique(stages[rows]):
        positions=np.flatnonzero(stages[rows]==stage);weights=desired[positions];weights/=weights.sum()
        count=min(len(positions),max(64,int(np.ceil(len(positions)/2))))
        selected.extend(rows[rng.choice(positions,count,replace=False,p=weights)].tolist())
        ess[str(stage)]=float(1/np.sum(weights**2))
    selected=np.array(selected,dtype=int)
    return selected,{'mix':mix,'state_density_ratios':ratio.tolist(),'source_cutoff_state_proportions':source.tolist(),
        'past_anchor_state_proportions':anchor.tolist(),'trusted_anchor_fraction':float(teacher.trusted.mean()),
        'effective_source_weights_by_stage':ess,'selected_source_cells':len(selected),
        'fit_max_stage':float(stages[selected].max()),'duplicates':int(len(selected)-len(np.unique(selected))),
        'scope':'Ratios from observed cutoff source and trusted challenge anchors. Fixed weights across past stages; unique seeded half-sample per stage. No future expression or annotation counts.'}
