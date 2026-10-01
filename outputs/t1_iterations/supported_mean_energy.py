"""Past-donor analytic minimum-energy mean field, followed by count-mass repair."""
import numpy as np
from forecast_transform_fidelity import implied_mass


def project(donors,desired,mapped,labels,supported):
    pred=donors.copy();records=[]
    for group in supported:
        rows=np.flatnonzero(labels==group)
        if not len(rows):continue
        a=np.asarray(donors[np.ix_(rows,mapped)],float)
        b=np.asarray(desired[np.ix_(rows,mapped)],float)
        support=a>0;counts=support.sum(0);delta=(b-a).mean(0)
        field=np.divide(len(rows)*delta,counts,out=np.zeros_like(delta),where=counts>0)
        minimum=float(np.sum(counts*field**2));budget=float(np.sum((b-a)**2))
        attenuation=min(1.,np.sqrt(budget/minimum)) if minimum else 1.
        raw=a+support*(attenuation*field)
        pred[np.ix_(rows,mapped)]=np.maximum(raw,0).astype(np.float32)
        records.append({'lineage':group,'cells':len(rows),'attenuation':float(attenuation),
                        'minimum_field_energy':minimum,'frozen_energy_budget':budget,
                        'unsupported_nonzero_mean_pairs':int(((counts==0)&(delta!=0)).sum()),
                        'clipped_entries':int((raw<0).sum())})
    changed=np.flatnonzero(np.isin(labels,supported));mass=implied_mass(donors,mapped)
    new_mass=implied_mass(pred,mapped)
    factors=np.divide(mass,new_mass,out=np.ones_like(mass),where=new_mass>0)
    for start in range(0,len(mapped),256):
        cols=mapped[start:start+256]
        block=np.expm1(np.asarray(pred[np.ix_(changed,cols)],float))
        pred[np.ix_(changed,cols)]=np.log1p(block*factors[changed,None]).astype(np.float32)
    for r in records:
        rows=np.flatnonzero(labels==r['lineage']);a=np.asarray(donors[np.ix_(rows,mapped)],float)
        b=np.asarray(desired[np.ix_(rows,mapped)],float);c=np.asarray(pred[np.ix_(rows,mapped)],float)
        energy=float(np.sum((c-a)**2));d=(b-a).mean(0);achieved=(c-a).mean(0)
        r.update(postrepair_energy=energy,energy_budget_passed=bool(energy<=r['frozen_energy_budget']+1e-8*max(1.,r['frozen_energy_budget'])),
                 relative_mean_error=float(np.linalg.norm(achieved-d)/np.linalg.norm(d)) if np.linalg.norm(d) else None)
    return pred,records


def numerical_controls():
    x=np.array([[0.,1.,2.,7.],[1.,0.,2.,8.]],np.float32);mapped=np.arange(3);labels=np.array(['fit','fallback'])
    y,records=project(x,x,mapped,labels,['fit'])
    np.testing.assert_allclose(x,y,atol=1e-6)
    desired=x.copy();desired[0,:3]+=.2
    y,records=project(x,desired,mapped,labels,['fit'])
    assert np.all(y[x==0]==0) and np.array_equal(y[1],x[1]) and np.array_equal(y[:,3],x[:,3])
    assert np.isfinite(y).all() and np.all(y>=0)
    np.testing.assert_allclose(implied_mass(y,mapped),implied_mass(x,mapped),rtol=1e-6)
    assert records[0]['unsupported_nonzero_mean_pairs']==1


if __name__=='__main__':
    numerical_controls();print('Identity, zero-lock, fallback, protected genes and mass controls passed')
