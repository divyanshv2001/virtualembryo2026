"""Past-only gene-pair variogram drift with guarded whole-cell resampling."""
import numpy as np
from kernel_population import KernelPopulation

class VariogramPopulation(KernelPopulation):
    def __init__(self,x,stages,cutoff,encoder,donors,base,base_indices,dimensions=500):
        if float(encoder['cutoff']) != cutoff or str(encoder['alignment']) != 'identity':
            raise ValueError('Require matching past-only encoder')
        if donors.shape != base.shape or len(base_indices) != len(donors): raise ValueError('Base mismatch')
        self.encoder=encoder; self.cutoff=cutoff; self.donors=donors; self.base=base
        self.base_indices=base_indices; self.features=encoder['official_features']
        self.trusted=encoder['trusted'][base_indices]
        rng=np.random.default_rng(2026092806)
        left=rng.integers(len(self.features),size=dimensions)
        right=(left+rng.integers(1,len(self.features),size=dimensions))%len(self.features)
        self.pairs=np.stack([left,right],axis=1)
        rows=np.flatnonzero(stages<=cutoff); self.times=np.unique(stages[rows])[-5:]
        if len(self.times)<5: raise ValueError('Five past stages required')
        means=[]
        for t in self.times:
            selected=rows[stages[rows]==t]; total=np.zeros(dimensions)
            for start in range(0,len(selected),256):
                values=np.asarray(x[np.ix_(selected[start:start+256],encoder['features'])],dtype=float)
                total += self.embed_values(values).sum(0)
            means.append(total/len(selected))
        self.stage_means=np.stack(means)
        self.anchor_mean=self.embed_official(donors).mean(0)
        self.candidate_features=self.embed_official(base)
        self.median_distance=0. # unused compatibility metadata; not a kernel bandwidth
    def embed_values(self,values):
        return np.sqrt(np.abs(values[:,self.pairs[:,0]]-values[:,self.pairs[:,1]]))/np.sqrt(len(self.pairs))
    def embed_official(self,values):
        return self.embed_values(values[:,self.features].astype(float))
    def predict(self,target,strength=.5,penalty=.01,window=5):
        prediction,origins,audit=super().predict(target,strength,penalty,window)
        audit['surrogate']='500 frozen past-feature gene pairs, expected absolute difference to power .5'
        audit['future_pairs_used']=False
        return prediction,origins,audit
    def save(self,path):
        np.savez_compressed(path,pairs=self.pairs,times=self.times,stage_means=self.stage_means,
            anchor_mean=self.anchor_mean,candidate_features=self.candidate_features,cutoff=np.array(self.cutoff))
