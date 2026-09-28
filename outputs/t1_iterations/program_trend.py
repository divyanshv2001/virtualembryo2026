"""Shared low-rank temporal profiles; statistical factors, not validated GEPs."""
import numpy as np
from annotation_trend import AnnotationTrend


class ProgramTrend(AnnotationTrend):
    def __init__(self,x,stages,source_labels,cutoff,donors,donor_labels,panel,symbols,features):
        super().__init__(x,stages,source_labels,cutoff,donors,donor_labels,panel,symbols,features)
        recent=np.unique(stages[stages<=cutoff])[-3:];source_labels=np.asarray(source_labels).astype(str)
        self.valid=np.zeros_like(self.positive_slope,dtype=bool);profiles=[]
        for k,label in enumerate(self.types):
            if self.support[k]<20:continue
            groups=[np.flatnonzero((stages==t)&(source_labels==label)) for t in recent]
            means=np.zeros((3,len(self.mapped)));counts=np.zeros_like(means)
            for start in range(0,len(self.mapped),512):
                genes=self.atlas[start:start+512]
                for j,rows in enumerate(groups):
                    a=np.expm1(np.asarray(x[np.ix_(rows,genes)],dtype=float));n=(a>0).sum(0)
                    mean=np.divide(a.sum(0),n,out=np.zeros(len(genes)),where=n>0)
                    means[j,start:start+len(genes)]=np.log(mean+.1)
                    counts[j,start:start+len(genes)]=n
            valid=counts.min(0)>=10;self.valid[k]=valid
            centered=means-means.mean(0)
            centered[:,~valid]=0
            # Equal type weight avoids inferring growth from captured cell counts.
            profiles.extend(centered)
        if not profiles:raise ValueError('No supported temporal profiles')
        matrix=np.stack(profiles)
        _,singular,basis=np.linalg.svd(matrix,full_matrices=False)
        rank=int((singular>max(singular[0]*1e-8,1e-10)).sum())
        self.basis=basis[:rank];self.singular_values=singular[:rank]
        self.original_positive_slope=self.positive_slope.copy()

    def predict_program(self,target,rank,detection=0.):
        if rank not in [2,4,8]:raise ValueError('Rank outside frozen pilot')
        rank_used=min(rank,len(self.basis));basis=self.basis[:rank_used]
        projected=(self.original_positive_slope@basis.T)@basis
        projected*=self.valid
        try:
            self.positive_slope=projected
            pred,indices,audit=super().predict(target,1.,detection,.02)
        finally:self.positive_slope=self.original_positive_slope
        return pred,indices,{**audit,'method':'shared_low_rank_temporal_projection',
            'rank_requested':rank,'rank_used':rank_used,
            'singular_values':self.singular_values.tolist(),
            'slope_relative_projection_error':float(np.linalg.norm(projected-self.original_positive_slope)/max(np.linalg.norm(self.original_positive_slope),1e-12)),
            'program_scope':'SVD of within-type centered positive-abundance profiles. Not cNMF, a single-cell activity decomposition or biologically validated programs.',
            'projection_assumption':'Past temporal directions share a low-rank gene subspace; projection can change individual gene signs.'}

    def save(self,path):
        super().save(path)
        with np.load(path) as source:fields={k:source[k] for k in source.files}
        np.savez_compressed(path,**fields,basis=self.basis,singular_values=self.singular_values,valid=self.valid)
