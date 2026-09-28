"""Conditional positive margins with four past-fitted states and fixed cell counts."""
import numpy as np
from detection_transfer import DetectionTransfer
from positive_quantile_forecast import PositiveQuantileForecast
from importance_source import IndexedRows
from robust_population import covariance_change


class StateQuantileForecast:
    def __init__(self,x,stages,cutoff,donors,panel,symbols,features):
        self.donors=donors;self.features=np.asarray(features)
        self.encoder=DetectionTransfer(x,stages,cutoff,donors,panel,symbols,states=4,
            alignment='identity',feature_scaling='unit',covariance_limit=.4)
        past=np.flatnonzero(stages<=cutoff);labels=self.encoder.model.clusterer.labels_
        self.models=[];self.skipped=[]
        recent=np.unique(stages[past])[-3:]
        for k in range(4):
            source=past[labels==k]
            anchors=np.flatnonzero((self.encoder.labels==k)&self.encoder.trusted)
            n=[int((stages[source]==t).sum()) for t in recent]
            if len(anchors)<20 or min(n)<20:
                self.skipped.append({'state':k,'anchors':len(anchors),'recent_source_counts':n});continue
            model=PositiveQuantileForecast(IndexedRows(x,source),stages[source],cutoff,
                donors[anchors],panel,symbols,features,min_positive=20)
            self.models.append((k,anchors,model))

    def predict(self,target,strength=1.,factor_cap=1.25,covariance_limit=.4):
        pred=self.donors.copy();audits=[]
        for state,anchors,model in self.models:
            values,_,audit=model.predict(target,strength,factor_cap,covariance_limit)
            pred[anchors]=values
            audits.append({'state':state,'anchors':len(anchors),'audit':audit})
        cov=covariance_change(self.donors[:,self.features],pred[:,self.features])
        rejected=cov>covariance_limit
        if rejected:pred=self.donors.copy()
        return pred,np.arange(len(pred)),{'method':'state_positive_quantile_drift','states':4,
            'strength':strength,'state_audits':audits,'skipped_states':self.skipped,
            'trusted_anchor_fraction':float(self.encoder.trusted.mean()),
            'covariance_change_before_global_guard':float(cov),'global_guard_rejected':bool(rejected),
            'covariance_change_vs_reference':0. if rejected else float(cov),
            'fixed_cell_counts':True,'zero_mask_preserved':bool(np.array_equal(pred==0,self.donors==0))}

    def save(self,path):
        self.encoder.save(path)
        for k,_,model in self.models:model.save(path.with_name(path.stem+f'_state{k}.npz'))
