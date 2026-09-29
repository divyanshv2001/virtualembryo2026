"""Past-only PCA transport couplings and ridge velocity extrapolation."""
from collections import Counter
import numpy as np
import torch
from sklearn.decomposition import PCA


from transport_latent_flow import log_transport,AffineTransportNet


class StageTransportNet(AffineTransportNet):
    def trajectory(self,z,times):
        history=[z]
        for previous,target in zip(times[:-1],times[1:]):
            span=float(target-previous);steps=max(1,int(np.ceil(span/.25)));dt=span/steps
            for step in range(steps):
                time_feature=float(previous+(step+.5)*dt)/.25
                ones=torch.ones((len(z),1),dtype=z.dtype)
                velocity=torch.cat([z,ones*time_feature,ones],1)@self.coefficient
                norm=torch.linalg.vector_norm(velocity,dim=1,keepdim=True)
                velocity*=torch.clamp(6/torch.clamp(norm,min=1e-6),max=1.)
                z=z+dt*velocity
            history.append(z)
        return torch.stack(history)


class StageTransportFlow:
    def __init__(self,x,stages,cutoff,panel,symbols,features,proxy_rows,proxy_values,mode,cell_budget=256,time_ridge=1.):
        if mode not in ['balanced','neutral','growth05','growth1']:raise ValueError('Unknown transport ablation')
        if time_ridge not in [None,1.,10.]:raise ValueError('Undeclared temporal regularization')
        counts=Counter(symbols);lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
        selected=np.array([lookup[panel[i]] for i in features]);rows=np.flatnonzero(stages<=cutoff)
        np.testing.assert_array_equal(rows,proxy_rows)
        values=np.asarray(x[np.ix_(rows,selected)],np.float32)
        self.center=values.mean(0);self.scale=np.maximum(values.std(0),.1)
        normalized=(values-self.center)/self.scale
        rng=np.random.default_rng(20260928);fit=rng.choice(len(rows),min(3000,len(rows)),replace=False)
        pca=PCA(n_components=min(16,len(features)),random_state=20260928).fit(normalized[fit])
        z=pca.transform(normalized).astype(float);times=np.unique(stages[rows])
        if len(times)<3 or not np.allclose(np.diff(times),.25):raise ValueError('Quarter-day historical stages required')
        stage_samples={t:np.sort(rng.choice(np.flatnonzero(stages[rows]==t),min(cell_budget,int((stages[rows]==t).sum())),replace=False)) for t in times}
        anchors=[];velocities=[];masses=[];couplings=[];fit_times=[]
        growth_power={'balanced':0.,'neutral':0.,'growth05':.5,'growth1':1.}[mode]
        proxy=(proxy_values-np.median(proxy_values))/max(float(proxy_values.std()),.1)
        for early,late in zip(times[:-1],times[1:]):
            ia=stage_samples[early];ib=stage_samples[late];za=z[ia];zb=z[ib]
            cost=((za[:,None]-zb[None,:])**2).sum(2);cost/=max(float(np.median(cost)),1e-8)
            a=np.exp(np.clip(growth_power*(late-early)*proxy[ia],-np.log(2.),np.log(2.)));a/=a.sum()
            b=np.full(len(ib),1/len(ib))
            coupling,audit=log_transport(cost,a,b,balanced=mode=='balanced')
            audit.update(source_stage=float(early),target_stage=float(late),source_cells=len(ia),target_cells=len(ib))
            couplings.append(audit)
            if not audit['converged']:raise ValueError('Historical transport did not converge: '+str(audit))
            mass=coupling.sum(1);destination=coupling@zb/np.maximum(mass[:,None],1e-12)
            anchors.append(za);velocities.append((destination-za)/(late-early));masses.append(mass)
            fit_times.append(np.full(len(za),((early+late)/2-cutoff)/.25))
        zfit=np.vstack(anchors);y=np.vstack(velocities);weight=np.concatenate(masses);weight/=weight.sum()
        h=np.column_stack([zfit,np.ones(len(zfit))]) if time_ridge is None else np.column_stack([zfit,np.concatenate(fit_times),np.ones(len(zfit))])
        penalty=np.eye(h.shape[1]);penalty[-1,-1]=1e-6
        if time_ridge is not None:penalty[-2,-2]=time_ridge
        coefficient=np.linalg.solve(h.T@(weight[:,None]*h)+penalty,h.T@(weight[:,None]*y))
        self.net=(AffineTransportNet if time_ridge is None else StageTransportNet)(pca.components_,pca.mean_,coefficient);self.net.eval()
        self.audit={'mode':mode,'fit_cells':len(rows),'fit_max_stage':float(stages[rows].max()),'pca_fit_cells':len(fit),
            'coupling_cell_budget_per_stage':cell_budget,'couplings':couplings,'growth_prior_power':growth_power,
            'method':'Past-only PCA transport velocities, explicit midpoint-stage affine field, bounded quarter-day extrapolation',
            'time_ridge':time_ridge,'time_feature':'(historical interval midpoint - cutoff)/.25; future solver uses its actual relative midpoint',
            'scope':'Time-conditioned affine ablation; not TrajectoryNet/CNF reproduction. Forecast time features lie outside training support. Entropy generalized scaling adaptation; not WOT/moscot reproduction. Growth priors are normalized bounded source weights, not calibrated birth/death rates. Future dynamics extrapolate an affine field learned from historical couplings. Every later endpoint used in a coupling is <=cutoff.'}

    def save(self,path):
        np.savez_compressed(path,center=self.center,scale=self.scale,basis=self.net.basis.numpy(),
            pca_center=self.net.pca_center.numpy(),coefficient=self.net.coefficient.numpy())
