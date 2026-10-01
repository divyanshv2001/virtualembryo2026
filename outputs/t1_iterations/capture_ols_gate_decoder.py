"""Frozen capture-sign and adjacent-interval gates; fixed additive decoder."""
import numpy as np
from forecast_transform_fidelity import implied_mass


def shifts(means,stages,ols,projected,omitted_ols):
    levels=np.unique(stages);sign=np.sign(ols)
    stable=(ols!=0)&np.all(np.sign(omitted_ols)==sign,axis=0)
    curvature=ols!=0
    for omission in [None,*range(len(means))]:
        keep=np.ones(len(means),bool)
        if omission is not None:keep[omission]=False
        if any(np.sum(keep&(stages==t))<2 for t in levels):raise ValueError('Mandatory omission unsupported')
        avg=[means[keep&(stages==t)].mean(0) for t in levels]
        curvature&=(np.sign(avg[1]-avg[0])==sign)&(np.sign(avg[2]-avg[1])==sign)
    return {'capture_OLS':ols,'capture_OLS_omission_gate':np.where(stable,ols,0),
            'capture_OLS_curvature_gate':np.where(curvature,ols,0),'capture_rank2':projected}


def decode(donors,mapped,labels,by_group):
    pred=donors.copy();clipped=0
    for group,shift in by_group.items():
        rows=np.flatnonzero(labels==group)
        for start in range(0,len(mapped),256):
            cols=mapped[start:start+256];raw=np.asarray(donors[np.ix_(rows,cols)],float)+shift[start:start+len(cols)]
            clipped+=int((raw<0).sum());pred[np.ix_(rows,cols)]=np.maximum(raw,0).astype(np.float32)
    changed=np.flatnonzero(np.isin(labels,list(by_group)));mass=implied_mass(donors,mapped);newmass=implied_mass(pred,mapped)
    factor=np.divide(mass,newmass,out=np.ones_like(mass),where=newmass>0)
    for start in range(0,len(mapped),256):
        cols=mapped[start:start+256];value=np.expm1(np.asarray(pred[np.ix_(changed,cols)],float))
        pred[np.ix_(changed,cols)]=np.log1p(value*factor[changed,None]).astype(np.float32)
    return pred,clipped


def controls():
    stages=np.repeat([7.5,7.75,8.],3);means=np.repeat([[0.,0.],[1.,3.],[2.,2.]],3,axis=0)
    ols=np.array([1.,1.]);s=shifts(means,stages,ols,ols,np.tile(ols,(9,1)))
    np.testing.assert_array_equal(s['capture_OLS_omission_gate'],ols)
    np.testing.assert_array_equal(s['capture_OLS_curvature_gate'],[1.,0.])
    a=np.array([[0.,1.,2.,7.],[1.,0.,2.,8.]],np.float32);mapped=np.arange(3);labels=np.array(['fit','fallback'])
    b,c=decode(a,mapped,labels,{'fit':np.array([.1,-.2,.3])})
    assert np.isfinite(b).all() and np.all(b>=0) and np.array_equal(b[1],a[1]) and np.array_equal(b[:,3],a[:,3])
    np.testing.assert_allclose(implied_mass(a,mapped),implied_mass(b,mapped),rtol=1e-6)
    b,c=decode(a,mapped,labels,{'fit':np.zeros(3)});np.testing.assert_allclose(a,b,atol=1e-6)


if __name__=='__main__':
    controls();print('Stable-sign/reversal, omission support, identity, mass, finite/protected/fallback controls passed')
