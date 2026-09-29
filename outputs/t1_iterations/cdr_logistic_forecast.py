"""Past-only logistic detection with observed detection-rate nuisance covariate."""
from collections import Counter
import numpy as np
import torch
from scipy.special import expit
from cdr_detection_forecast import CDRDetectionForecast
from logistic_detection_head import fit_logistic


class CDRLogisticForecast(CDRDetectionForecast):
    def __init__(self,*args,detection_ridge=1.,emit=lambda **kw:None,saved_heads=None,**kwargs):
        super().__init__(*args,cdr_ridge=1.,**kwargs)
        if saved_heads is not None:
            saved=np.load(saved_heads)
            np.testing.assert_array_equal(saved['donor_cdr'],self.donor_cdr)
            if float(saved['cdr_mean'])!=self.cdr_mean or float(saved['cdr_scale'])!=self.cdr_scale:
                raise ValueError('Saved CDR normalization mismatch')
            coef=saved['coef']
            if coef.shape!=(len(self.zcenter)+2,len(self.mapped)) or not np.isfinite(coef).all():
                raise ValueError('Invalid saved logistic coefficients')
            self.cdr_logistic_coef=coef.copy()
            self.audit.update(detection_link='logistic',detection_ridge=detection_ridge,
                              fitted_head_reused=str(saved_heads),additional_logistic_updates=0,
                              scope='Reuse fitted head from preserved failed run; runner verifies training inputs and original module provenance. Original optimizer diagnostics remain in failed-run events.')
            return
        x,stages,cutoff=args[:3];panel,symbols=args[4:6]
        counts=Counter(symbols);lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
        rows=np.flatnonzero(stages<=cutoff)
        atlas=np.array([lookup[panel[i]] for i in self.mapped]);feat=[lookup[panel[i]] for i in self.features]
        cdr_count=np.zeros(len(rows))
        for start in range(0,len(atlas),512):
            cdr_count+=(np.asarray(x[np.ix_(rows,atlas[start:start+512])])>0).sum(1)
        with torch.no_grad():
            h=self.net.encode(torch.tensor((np.asarray(x[np.ix_(rows,feat)])-self.center)/self.scale))[0].numpy().astype(float)-self.zcenter
        matrix=np.column_stack([h,(cdr_count/len(atlas)-self.cdr_mean)/self.cdr_scale])
        self.cdr_logistic_coef=np.zeros((matrix.shape[1]+1,len(atlas)));diagnostics=[]
        for start in range(0,len(atlas),512):
            sl=slice(start,min(start+512,len(atlas)))
            y=(np.asarray(x[np.ix_(rows,atlas[sl])])>0).astype(float)
            coef,audit=fit_logistic(matrix,y,detection_ridge)
            self.cdr_logistic_coef[:,sl]=coef;diagnostics.append(audit);emit(block_start=start,**audit)
        self.audit.update(detection_link='logistic',detection_ridge=detection_ridge,logistic_diagnostics=diagnostics,
                          scope='MAST-inspired observed CDR nuisance covariate with L2 logistic detection, not Bayesian MAST inference. Compare archived non-CDR logistic at identical ridge. Flow/positive head fixed, donor CDR held constant.')

    def detection_probability(self,h,sl):
        matrix=np.column_stack([h,self.donor_cdr])
        return np.clip(expit(matrix@self.cdr_logistic_coef[:-1,sl]+self.cdr_logistic_coef[-1,sl]),1e-4,1-1e-4)
