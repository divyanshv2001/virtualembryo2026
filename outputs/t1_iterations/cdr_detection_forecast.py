"""Detection-rate nuisance adjustment with a frozen positive-expression head."""
from collections import Counter
import numpy as np
import torch
from feature_panel_forecast import FeaturePanelForecast


class CDRDetectionForecast(FeaturePanelForecast):
    def __init__(self,*args,cdr_ridge=1.,**kwargs):
        if cdr_ridge not in [.1,1.]:raise ValueError('Undeclared CDR regularization')
        super().__init__(*args,**kwargs)
        x,stages,cutoff=args[:3];panel,symbols=args[4:6]
        rows=np.flatnonzero(stages<=cutoff)
        counts=Counter(symbols);lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
        atlas=np.array([lookup[panel[i]] for i in self.mapped])
        detected_counts=np.zeros(len(rows))
        for start in range(0,len(atlas),512):
            detected_counts+=(np.asarray(x[np.ix_(rows,atlas[start:start+512])])>0).sum(1)
        cdr=detected_counts/len(atlas)
        self.cdr_mean=float(cdr.mean());self.cdr_scale=max(float(cdr.std()),1e-3)
        self.donor_cdr=((self.donors[:,self.mapped]>0).mean(1)-self.cdr_mean)/self.cdr_scale
        feat=[lookup[panel[i]] for i in self.features]
        with torch.no_grad():
            h=self.net.encode(torch.tensor((np.asarray(x[np.ix_(rows,feat)])-self.center)/self.scale))[0].numpy().astype(float)-self.zcenter
        matrix=np.column_stack([h,(cdr-self.cdr_mean)/self.cdr_scale]);n=len(matrix)
        gram=matrix.T@matrix/n+np.diag([1.]*h.shape[1]+[cdr_ridge])
        self.cdr_coef=np.zeros((matrix.shape[1],len(atlas)))
        for start in range(0,len(atlas),512):
            sl=slice(start,min(start+512,len(atlas)))
            detected=(np.asarray(x[np.ix_(rows,atlas[sl])])>0).astype(float)
            self.cdr_coef[:,sl]=np.linalg.solve(gram,matrix.T@detected/n)
        self.cdr_coef[:,~self.support]=0
        self.audit.update(cdr_mean=self.cdr_mean,cdr_scale=self.cdr_scale,cdr_ridge=cdr_ridge,
                          cdr_gene_count=len(atlas),cdr_fit_rows=len(rows),cdr_fit_max_stage=float(stages[rows].max()),
                          cdr_policy='Fraction detected among exact unique mapped genes; hold observed donor CDR covariate fixed during forecasting',
                          scope='Own linear-ridge CDR adjustment inspired by MAST nuisance covariate. Not logistic/Bayesian MAST inference. Positive head and flow fixed; CDR can contain biological signal and part-whole dependence.')

    def detection_probability(self,h,sl):
        matrix=np.column_stack([h,self.donor_cdr])
        return np.clip(self.pmean[sl]+matrix@self.cdr_coef[:,sl],1e-4,1-1e-4)
