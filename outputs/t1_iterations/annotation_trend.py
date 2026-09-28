"""Past-only within-annotation detection and positive-abundance trends."""
from collections import Counter
import numpy as np
from robust_population import stable_slope, covariance_change


class AnnotationTrend:
    def __init__(self,x,stages,source_labels,cutoff,donors,donor_labels,panel,symbols,features,min_source_cells=20,min_positive_cells=10):
        if not 20<=min_source_cells<=200 or not 10<=min_positive_cells<=100:
            raise ValueError('Support thresholds outside frozen bounds')
        self.min_source_cells=min_source_cells;self.min_positive_cells=min_positive_cells
        self.cutoff=cutoff;self.donors=donors;self.donor_labels=np.asarray(donor_labels).astype(str)
        self.features=np.asarray(features);source_labels=np.asarray(source_labels).astype(str)
        recent=np.unique(stages[stages<=cutoff])[-3:]
        if len(recent)!=3:raise ValueError('Three observed stages required')
        counts=Counter(symbols);lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
        self.mapped=np.array([i for i,s in enumerate(panel) if s in lookup])
        self.atlas=np.array([lookup[panel[i]] for i in self.mapped])
        self.types=sorted(set(source_labels[stages<=cutoff]))
        shape=(len(self.types),len(self.mapped))
        self.positive_slope=np.zeros(shape);self.detection_slope=np.zeros(shape)
        self.support=np.zeros(len(self.types),int)
        for k,label in enumerate(self.types):
            groups=[np.flatnonzero((stages==t)&(source_labels==label)) for t in recent]
            support=min(map(len,groups));self.support[k]=support
            if support<min_source_cells:continue
            for start in range(0,len(self.mapped),256):
                genes=self.atlas[start:start+256];means=[];errors=[];probs=[];perrors=[];positive_counts=[]
                for rows in groups:
                    a=np.expm1(np.asarray(x[np.ix_(rows,genes)],dtype=float));det=a>0;n=det.sum(0)
                    positive_counts.append(n)
                    total=a.sum(0);mean=np.divide(total,n,out=np.zeros_like(total),where=n>0)
                    second=np.divide((a*a).sum(0),n,out=np.zeros_like(total),where=n>0)
                    var=np.maximum(second-mean*mean,0)
                    means.append(np.log(mean+.1));errors.append(np.sqrt(var/np.maximum(n,1))/(mean+.1))
                    p=(n+1)/(len(rows)+2);probs.append(p);perrors.append(np.sqrt(p*(1-p)/len(rows)))
                valid=np.min(np.stack(positive_counts),axis=0)>=min_positive_cells
                shrink=support/(support+64)
                self.positive_slope[k,start:start+len(genes)]=stable_slope(recent,np.stack(means),np.stack(errors))*valid*shrink
                self.detection_slope[k,start:start+len(genes)]=stable_slope(recent,np.stack(probs),np.stack(perrors))*shrink

    def predict(self,target,expression=1.,detection=.5,cap=.02):
        if target<=self.cutoff or not 0<=expression<=1 or not 0<=detection<=1 or not 0<=cap<=.05:
            raise ValueError('Invalid frozen parameters')
        horizon=target-self.cutoff
        old_mass=np.expm1(self.donors[:,self.mapped].astype(float)).sum(1)
        pred=self.donors.copy();used=0.;details={}
        for backoff in [1.,.5,.25,.125,0.]:
            candidate=self.donors.copy();added=deleted=0
            for k,label in enumerate(self.types):
                rows=np.flatnonzero(self.donor_labels==label)
                if len(rows)<20 or self.support[k]<self.min_source_cells:continue
                factors=np.exp(np.clip(horizon*expression*backoff*self.positive_slope[k],-np.log(1.25),np.log(1.25)))
                candidate[np.ix_(rows,self.mapped)]=np.log1p(np.expm1(self.donors[np.ix_(rows,self.mapped)].astype(float))*factors).astype(np.float32)
                changes=np.clip(horizon*detection*backoff*self.detection_slope[k],-cap,cap)
                for j in np.flatnonzero(np.abs(changes)*len(rows)>=.5):
                    gene=self.mapped[j];n=int(np.floor(abs(changes[j])*len(rows)+.5))
                    values=candidate[rows,gene];positive=rows[values>0];zero=rows[values==0]
                    rng=np.random.default_rng(np.random.SeedSequence([20260928,k,int(gene),31]))
                    if changes[j]>0 and len(positive) and len(zero):
                        chosen=rng.permutation(zero)[:n]
                        candidate[chosen,gene]=candidate[rng.choice(positive,len(chosen),replace=True),gene];added+=len(chosen)
                    elif changes[j]<0 and len(positive):
                        chosen=rng.permutation(positive)[:n];candidate[chosen,gene]=0;deleted+=len(chosen)
            a=np.expm1(candidate[:,self.mapped].astype(float));total=a.sum(1)
            empty=(total==0)&(old_mass>0)
            a[empty]=np.expm1(self.donors[np.ix_(empty,self.mapped)].astype(float));total=a.sum(1)
            ratio=np.divide(old_mass,total,out=np.ones_like(old_mass),where=total>0)
            candidate[:,self.mapped]=np.log1p(a*ratio[:,None]).astype(np.float32)
            cov=covariance_change(self.donors[:,self.features],candidate[:,self.features])
            if cov<=.4:
                pred=candidate;used=backoff;details={'attempted_additions':added,'attempted_deletions':deleted};break
        return pred,np.arange(len(pred)),{**details,'method':'annotation_positive_detection_trend',
            'requested_expression':expression,'requested_detection':detection,'backoff':used,
            'covariance_change_vs_reference':float(cov),'type_support':dict(zip(self.types,self.support.tolist())),
            'minimum_source_cells':self.min_source_cells,'minimum_positive_cells':self.min_positive_cells,
            'supported_donor_fraction':float(np.mean(np.isin(self.donor_labels,np.asarray(self.types)[self.support>=self.min_source_cells]))),
            'actual_additions':int(((self.donors==0)&(pred>0)).sum()),
            'actual_deletions':int(((self.donors>0)&(pred==0)).sum()),
            'fixed_cell_counts':True,'assumption':'Within-type historical abundance/detection trends persist; annotation strings are not measured lineage links.'}

    def save(self,path):
        np.savez_compressed(path,positive_slope=self.positive_slope,detection_slope=self.detection_slope,
            mapped=self.mapped,atlas=self.atlas,support=self.support,types=np.asarray(self.types),cutoff=self.cutoff,
            min_source_cells=self.min_source_cells,min_positive_cells=self.min_positive_cells)
