"""Past-only latent-dimensionality sensitivity with fixed coupling samples."""
from collections import Counter
import numpy as np
from sklearn.decomposition import PCA
from transport_latent_flow import log_transport,AffineTransportNet

class DimensionalTransportFlow:
    def __init__(self,x,stages,cutoff,panel,symbols,features,proxy_rows,proxy_values,mode,cell_budget=256,pca_budget=3000,latent_dim=16):
        if latent_dim not in [5,8,16,24]:raise ValueError('Undeclared latent dimension')
        if cell_budget not in [16,256,512,1024] or pca_budget not in [None,3000]:raise ValueError('Undeclared resolution budget')
        if mode not in ['balanced','neutral','growth05','growth1']:raise ValueError('Unknown transport ablation')
        counts=Counter(symbols);lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
        selected=np.array([lookup[panel[i]] for i in features]);rows=np.flatnonzero(stages<=cutoff)
        np.testing.assert_array_equal(rows,proxy_rows)
        values=np.asarray(x[np.ix_(rows,selected)],np.float32)
        self.center=values.mean(0);self.scale=np.maximum(values.std(0),.1)
        normalized=(values-self.center)/self.scale
        rng=np.random.default_rng(20260928);baseline_fit=rng.choice(len(rows),min(3000,len(rows)),replace=False)
        fit=np.arange(len(rows)) if pca_budget is None else baseline_fit
        pca=PCA(n_components=min(latent_dim,len(features)),random_state=20260928).fit(normalized[fit])
        z=pca.transform(normalized).astype(float);times=np.unique(stages[rows])
        if len(times)<3 or not np.allclose(np.diff(times),.25):raise ValueError('Quarter-day historical stages required')
        stage_samples={t:np.sort(rng.choice(np.flatnonzero(stages[rows]==t),min(cell_budget,int((stages[rows]==t).sum())),replace=False)) for t in times}
        self.pca_rows=rows[fit];self.sample_rows={str(t):rows[index] for t,index in stage_samples.items()}
        anchors=[];velocities=[];masses=[];couplings=[]
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
        zfit=np.vstack(anchors);y=np.vstack(velocities);weight=np.concatenate(masses);weight/=weight.sum()
        h=np.column_stack([zfit,np.ones(len(zfit))]);penalty=np.eye(h.shape[1]);penalty[-1,-1]=1e-6
        coefficient=np.linalg.solve(h.T@(weight[:,None]*h)+penalty,h.T@(weight[:,None]*y))
        self.net=AffineTransportNet(pca.components_,pca.mean_,coefficient);self.net.eval()
        self.audit={'mode':mode,'fit_cells':len(rows),'fit_max_stage':float(stages[rows].max()),'pca_fit_cells':len(fit),'pca_budget':pca_budget,'requested_latent_dimensions':latent_dim,'actual_latent_dimensions':pca.n_components_,
            'coupling_cell_budget_per_stage':cell_budget,'couplings':couplings,'growth_prior_power':growth_power,
            'method':'Past-only PCA transport barycentric velocities, ridge1 affine field, quarter-day Euler extrapolation',
            'scope':'Latent-dimensionality ablation on the prepared cohort; not full raw-atlas training. PCA-all preserves the baseline stage sampling RNG stream. Entropy generalized scaling adaptation; not WOT/moscot reproduction. Growth priors are normalized bounded source weights, not calibrated birth/death rates. Future dynamics extrapolate an affine field learned from historical couplings. Every later endpoint used in a coupling is <=cutoff.'}

    def save(self,path):
        np.savez_compressed(path,center=self.center,scale=self.scale,basis=self.net.basis.numpy(),
            pca_center=self.net.pca_center.numpy(),coefficient=self.net.coefficient.numpy(),
            pca_source_rows=self.pca_rows,**{f"stage_{t}_rows":r for t,r in self.sample_rows.items()})
