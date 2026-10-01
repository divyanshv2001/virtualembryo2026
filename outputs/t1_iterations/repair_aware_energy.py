"""Donor-only monotone count interpolation with a frozen log-energy budget."""
import numpy as np
from forecast_transform_fidelity import implied_mass


def attenuate(donors,projected,mapped,labels,records):
    output=projected.copy();diagnostics=[]
    for record in records:
        rows=np.flatnonzero(labels==record['lineage'])
        a=np.asarray(donors[np.ix_(rows,mapped)],float)
        b=np.asarray(projected[np.ix_(rows,mapped)],float)
        counts=np.expm1(a);difference=np.expm1(b)-counts
        budget=record['frozen_energy_budget'];limit=budget+1e-8*max(1.,budget)
        def trial(t):
            value=np.log1p(np.maximum(counts+t*difference,0)).astype(np.float32)
            energy=float(np.sum((np.asarray(value,float)-a)**2))
            return value,energy
        value,energy=trial(1.);lo=1.;steps=0
        if energy>limit:
            lo=0.;hi=1.
            for _ in range(40):
                mid=(lo+hi)/2;_,tested=trial(mid)
                if tested<=limit:lo=mid
                else:hi=mid
            value,energy=trial(lo);steps=40
        if energy>limit:
            lo=0.;value=donors[np.ix_(rows,mapped)].copy();energy=0.
        # Preserve unchanged endpoints bitwise, including rows outside fitted lineages.
        if lo==1.:value=projected[np.ix_(rows,mapped)].copy()
        output[np.ix_(rows,mapped)]=value
        diagnostics.append({**record,'interpolation_amplitude':lo,'bisection_steps':steps,
                            'final_energy':energy,'final_energy_budget_passed':bool(energy<=limit)})
    return output,diagnostics


def numerical_controls():
    a=np.array([[0.,1.,2.,7.],[1.,0.,2.,8.]],np.float32);mapped=np.arange(3)
    counts=np.expm1(np.asarray(a,float));b=counts.copy();b[0,1]*=.5;b[0,2]+=counts[0,1]*.5
    b=np.log1p(b).astype(np.float32);b[:,3]=a[:,3];b[1]=a[1]
    labels=np.array(['fit','fallback']);records=[{'lineage':'fit','frozen_energy_budget':.01}]
    y,d=attenuate(a,b,mapped,labels,records)
    assert 0<d[0]['interpolation_amplitude']<1 and d[0]['final_energy_budget_passed']
    assert np.all(y[a==0]==0) and np.array_equal(y[1],a[1]) and np.array_equal(y[:,3],a[:,3])
    np.testing.assert_allclose(implied_mass(y,mapped),implied_mass(a,mapped),rtol=1e-6)
    y,d=attenuate(a,a,mapped,labels,[{'lineage':'fit','frozen_energy_budget':0.}])
    assert np.array_equal(y,a) and d[0]['interpolation_amplitude']==1.


if __name__=='__main__':
    numerical_controls();print('Count interpolation identity, support, mass, protected/fallback and emitted-energy controls passed')
