"""Bounded cell-level NMF consensus and past-only usage dynamics adaptation."""
import warnings
import numpy as np
from sklearn.decomposition import NMF, non_negative_factorization
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from annotation_trend import AnnotationTrend
from robust_population import stable_slope


class CellProgramTrend(AnnotationTrend):
    def __init__(self,x,stages,source_labels,cutoff,donors,donor_labels,panel,symbols,features,rank=8,nmf_max_iter=200,gene_budget=384):
        super().__init__(x,stages,source_labels,cutoff,donors,donor_labels,panel,symbols,features)
        if rank not in [8,16]:raise ValueError('Rank outside frozen pilot')
        if not 100<=nmf_max_iter<=1000:raise ValueError('Iteration budget outside bounded pilot')
        if gene_budget not in [384,2048]:raise ValueError('Gene budget outside frozen coverage pilot')
        past=np.flatnonzero(stages<=cutoff)
        fitrows=np.sort(np.random.default_rng(20260928).choice(past,min(3000,len(past)),replace=False))
        lookup={int(o):int(a) for o,a in zip(self.mapped,self.atlas)}
        self.nmf_features=np.array([lookup[int(g)] for g in self.features])
        if gene_budget!=384:
            # Same logged-expression variance and stable atlas-order ties as PopulationForecast.
            eligible=np.sort(self.atlas);variance=np.zeros(len(eligible))
            for start in range(0,len(eligible),512):
                block=eligible[start:start+512]
                variance[start:start+len(block)]=np.asarray(x[np.ix_(past,block)],dtype=float).var(0)
            self.nmf_features=np.sort(eligible[np.argsort(-variance,kind='stable')[:gene_budget]])
        raw=np.expm1(np.asarray(x[np.ix_(fitrows,self.nmf_features)],dtype=float))
        self.nmf_scale=np.maximum(raw.std(0),.1);training=raw/self.nmf_scale
        rank=min(rank,training.shape[1],training.shape[0]-1)
        replicas=[];iterations=[];warning_count=0
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always',ConvergenceWarning)
            for seed in [20260928,20260929,20260930]:
                model=NMF(n_components=rank,init='random',random_state=seed,max_iter=nmf_max_iter,tol=1e-3)
                model.fit_transform(training);h=model.components_
                replicas.append(h/np.maximum(np.linalg.norm(h,axis=1,keepdims=True),1e-12));iterations.append(int(model.n_iter_))
            warning_count=sum(issubclass(w.category,ConvergenceWarning) for w in recorded)
        stack=np.concatenate(replicas)
        labels=KMeans(n_clusters=rank,n_init=10,random_state=20260928).fit_predict(stack)
        consensus=np.stack([np.median(stack[labels==k],axis=0) for k in range(rank)])
        self.consensus=consensus/np.maximum(consensus.sum(1,keepdims=True),1e-12)
        self.cluster_sizes=np.bincount(labels,minlength=rank)
        self.consensus_dispersion=np.array([np.linalg.norm(stack[labels==k]-consensus[k],axis=1).mean() for k in range(rank)])
        usages=[];usage_iterations=[]
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always',ConvergenceWarning)
            for start in range(0,len(past),1000):
                values=np.expm1(np.asarray(x[np.ix_(past[start:start+1000],self.nmf_features)],dtype=float))/self.nmf_scale
                w,_,niter=non_negative_factorization(values,W=np.ones((len(values),rank)),H=self.consensus.copy(),
                    n_components=rank,init='custom',update_H=False,solver='mu',max_iter=200,tol=1e-3)
                w/=np.maximum(w.sum(1,keepdims=True),1e-12);usages.append(w);usage_iterations.append(int(niter))
            warning_count+=sum(issubclass(w.category,ConvergenceWarning) for w in recorded)
        usages=np.concatenate(usages)
        train_usage=usages[np.searchsorted(past,fitrows)]
        self.usage_center=train_usage.mean(0);self.usage_scale=np.maximum(train_usage.std(0),.05)
        z=(usages-self.usage_center)/self.usage_scale
        zfit=z[np.searchsorted(past,fitrows)];gram=zfit.T@zfit/len(zfit)+np.eye(rank)
        self.decoder=np.zeros((rank,len(self.mapped)))
        for start in range(0,len(self.mapped),512):
            genes=self.atlas[start:start+512]
            raw_block=np.asarray(x[np.ix_(fitrows,genes)],dtype=float)
            y=np.log(np.expm1(raw_block)+.1)
            coef=np.linalg.solve(gram,zfit.T@y/len(zfit))
            coef[:,(raw_block>0).sum(0)<20]=0
            self.decoder[:,start:start+len(genes)]=coef
        recent=np.unique(stages[past])[-3:];source_labels=np.asarray(source_labels).astype(str)
        self.usage_slope=np.zeros((len(self.types),rank))
        for k,label in enumerate(self.types):
            groups=[np.flatnonzero((stages[past]==t)&(source_labels[past]==label)) for t in recent]
            n=min(map(len,groups))
            if n<20:continue
            means=np.stack([z[g].mean(0) for g in groups])
            errors=np.stack([z[g].std(0)/np.sqrt(len(g)) for g in groups])
            self.usage_slope[k]=stable_slope(recent,means,errors)*n/(n+64)
        self.positive_slope=self.usage_slope@self.decoder
        self.fit_audit={'rank':rank,'fit_cells':len(fitrows),'features':len(self.nmf_features),
            'requested_gene_budget':gene_budget,'feature_selection':'Past-only logged-expression variance; unique exact-mapped genes; covariance guard retains its original features.',
            'nmf_max_iter':nmf_max_iter,'nmf_tolerance':1e-3,
            'replicate_iterations':iterations,'usage_iterations':usage_iterations,'convergence_warnings':warning_count,
            'consensus_cluster_sizes':self.cluster_sizes.tolist(),'consensus_dispersion':self.consensus_dispersion.tolist(),
            'adaptation':'Three NMF replicas, median consensus, fixed-component multiplicative usage fitting; normalized abundance instead of raw counts, variance-selected genes instead of paper v-score selection, no component outlier filter. Ridge1 full-panel log-abundance decoder and within-type usage slopes are added forecasting hypotheses, not cNMF reproduction.',
            'scope':'Cell-level factors are not validated biological activity programs; convergence and consensus with three replicas do not establish recovery or temporal accuracy.'}

    def predict_cell_program(self,target,detection=0.):
        pred,indices,audit=super().predict(target,1.,detection,.02)
        return pred,indices,{**audit,'method':'cell_nmf_consensus_usage_drift','fit_audit':self.fit_audit}

    def save(self,path):
        super().save(path)
        with np.load(path) as source:fields={k:source[k] for k in source.files}
        np.savez_compressed(path,**fields,consensus=self.consensus,nmf_features=self.nmf_features,nmf_scale=self.nmf_scale,
            decoder=self.decoder,usage_slope=self.usage_slope,usage_center=self.usage_center,usage_scale=self.usage_scale,
            cluster_sizes=self.cluster_sizes,consensus_dispersion=self.consensus_dispersion)
