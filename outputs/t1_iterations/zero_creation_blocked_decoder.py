"""Fixed log-shift ablation: donor zeros remain zero; positives may be clipped away."""
import numpy as np
from forecast_transform_fidelity import implied_mass


def decode_blocked(donors,mapped,donor_labels,saved,supported,name):
    pred=donors.copy()
    for group in supported:
        shift=saved[group+'__'+name]-saved[group+'__persistence'];rows=np.flatnonzero(donor_labels==group)
        for start in range(0,len(mapped),256):
            cols=mapped[start:start+256];a=donors[np.ix_(rows,cols)]
            block=np.asarray(a,float)+shift[start:start+len(cols)]
            pred[np.ix_(rows,cols)]=np.where(a>0,np.maximum(block,0),0).astype(np.float32)
    changed_rows=np.flatnonzero(np.isin(donor_labels,supported));mass=implied_mass(donors,mapped);changed_mass=implied_mass(pred,mapped)
    factors=np.divide(mass,changed_mass,out=np.ones_like(mass),where=changed_mass>0)
    for start in range(0,len(mapped),256):
        cols=mapped[start:start+256];a=np.expm1(np.asarray(pred[np.ix_(changed_rows,cols)],float))
        pred[np.ix_(changed_rows,cols)]=np.log1p(a*factors[changed_rows,None]).astype(np.float32)
    return pred


def numerical_controls():
    x=np.array([[0.,1.,2.,7.],[1.,0.,2.,8.]],np.float32);cols=np.arange(3);labels=np.array(['a','fallback'])
    saved={'a__persistence':np.zeros(3),'a__capture_quarter':np.array([1.,-.2,.3])}
    y=decode_blocked(x,cols,labels,saved,['a'],'capture_quarter')
    assert np.all(y[x==0]==0) and np.array_equal(y[1],x[1]) and np.array_equal(y[:,3],x[:,3])
    np.testing.assert_allclose(implied_mass(y,cols),implied_mass(x,cols),rtol=1e-6)
    assert np.isfinite(y).all() and np.all(y>=0)
    saved['a__capture_quarter']=np.zeros(3)
    np.testing.assert_allclose(decode_blocked(x,cols,labels,saved,['a'],'capture_quarter'),x,atol=1e-6)


if __name__=='__main__':
    numerical_controls();print('Blocked decoder zero-lock, fallback, protected genes, mass and identity controls passed')
