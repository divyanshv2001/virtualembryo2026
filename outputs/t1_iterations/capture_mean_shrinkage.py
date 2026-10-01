"""Fixed diagonal normal-prior regularizer; capture dispersion is not embryo SE."""
import numpy as np


def forecast(previous, current, horizon_ratio=1.):
    a,b=np.asarray(previous,float),np.asarray(current,float)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1] or min(len(a),len(b))<2:
        raise ValueError('Require two captures at each stage in the same gene basis')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or horizon_ratio<=0:
        raise ValueError('Invalid capture means or horizon')
    mean_a,mean_b=a.mean(0),b.mean(0)
    delta=mean_b-mean_a
    variance=a.var(0,ddof=1)/len(a)+b.var(0,ddof=1)/len(b)
    tau2=max(float(np.mean(delta*delta-variance)),0.)
    gain=np.divide(tau2,tau2+variance,out=np.zeros_like(delta),where=tau2+variance>0)
    return {'persistence':mean_b,'capture_linear':mean_b+horizon_ratio*delta,
            'capture_quarter':mean_b+.25*horizon_ratio*delta,
            'capture_shrunk':mean_b+horizon_ratio*gain*delta}, {'tau2':tau2,
            'gain_quantiles':np.quantile(gain,[0,.25,.5,.75,1]).tolist()}
