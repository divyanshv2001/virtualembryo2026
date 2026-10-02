"""Past-only full mapped-gene joint training and anchored neural decoder adaptation."""
import copy
import hashlib
import numpy as np
import torch
import anndata as ad
from scipy import sparse
from scnode_joint_model import JointModel,joint_loss
from feature_panel_forecast import FeaturePanelForecast


def batch(x,rows,columns):
    return torch.tensor(np.asarray(x[np.ix_(rows,columns)],dtype=np.float32))


def train_pair(x,stages,columns,cutoff,run,emit):
    eligible=np.flatnonzero(stages<=cutoff)
    if not len(eligible): raise ValueError('No past fit rows')
    times=np.unique(stages[eligible]);groups=[np.flatnonzero(stages==t) for t in times]
    if times.max()>cutoff: raise ValueError('Future fit rows')
    torch.manual_seed(20261002);rng=np.random.default_rng(20261002)
    model=JointModel(len(columns))
    optimizer=torch.optim.Adam(list(model.encoder.parameters())+list(model.mu.parameters())+list(model.std.parameters())+list(model.decoder.parameters()),lr=.001,betas=(.95,.99))
    for step in range(200):
        group=groups[int(rng.integers(len(groups)))];values=batch(x,rng.choice(group,64,replace=True),columns)
        latent=model.sample(values,torch.randn(64,32));loss=((model.decoder(latent)-values)**2).mean()
        if not torch.isfinite(loss): raise ValueError('Nonfinite biological pretrain')
        optimizer.zero_grad();loss.backward();optimizer.step()
        if (step+1)%50==0: emit('vae_pretraining',step=step+1,loss=float(loss.detach()))
    initial=copy.deepcopy(model.state_dict());torch.save(initial,run/'shared_pretrain.pt')
    flows={};matched_stream=None
    for beta in [0.,.1]:
        name='scnode_beta_'+str(beta)
        model.load_state_dict(initial);torch.manual_seed(20261002);rng=np.random.default_rng(20261002)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001,betas=(.95,.99));history=[];stream=hashlib.sha256()
        for step in range(1000):
            batch_rows=[rng.choice(g,32,replace=True) for g in groups]
            target_rows=[rng.choice(g,200,replace=True) for g in groups]
            for rows in batch_rows+target_rows:stream.update(rows.tobytes())
            batches=[batch(x,rows,columns) for rows in batch_rows]
            targets=[batch(x,rows,columns) for rows in target_rows]
            noises=[torch.randn(32,32) for _ in range(len(groups)+1)]
            for noise in noises:stream.update(noise.numpy().tobytes())
            loss,expression,dynamic=joint_loss(model,batches,torch.tensor(times-times[0],dtype=torch.float32),noises,beta,targets=targets)
            if not torch.isfinite(loss): raise ValueError('Nonfinite biological joint loss')
            optimizer.zero_grad();loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()): raise ValueError('Nonfinite biological gradient')
            optimizer.step()
            if (step+1)%25==0:
                record={'candidate':name,'step':step+1,'loss':float(loss.detach()),'expression_loss':float(expression.detach()),'dynamic_loss':float(dynamic.detach())};history.append(record)
                torch.save({'net':model.state_dict(),'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'numpy_rng':rng.bit_generator.state,'history':history,'beta':beta,'fit_max_stage':float(times.max()),'seed':20261002,'steps':1000},run/(name+'.pt'));emit('scnode_training_checkpoint',**record)
        if matched_stream is not None and stream.hexdigest()!=matched_stream: raise ValueError('New arm random streams differ')
        matched_stream=stream.hexdigest();emit('matched_training_stream',candidate=name,draw_sha256=matched_stream)
        flows[name]=copy.deepcopy(model).eval()
    return flows


class MeanTrajectory:
    def __init__(self,model):self.model=model
    def encode(self,x):return self.model.encode(x)
    def trajectory(self,z,times):return self.model.trajectory(z,times).transpose(0,1)


class AnchoredNeuralForecast(FeaturePanelForecast):
    def __init__(self,x,stages,cutoff,donors,mapped,columns,model,guard_features,anchor_path):
        self.cutoff=cutoff;self.donors=donors;self.mapped=np.asarray(mapped);self.features=self.mapped
        self.center=np.zeros(len(mapped),dtype=np.float32);self.scale=np.ones(len(mapped),dtype=np.float32)
        self.guard_features=guard_features;self.model=model;self.net=MeanTrajectory(model)
        rows=np.flatnonzero(stages<=cutoff);z=[]
        with torch.no_grad():
            for start in range(0,len(rows),128):z.append(model.encode(batch(x,rows[start:start+128],columns))[0].numpy())
        z=np.concatenate(z).astype(float);self.zcenter=z.mean(0);h=z-self.zcenter
        gram=h.T@h/len(h)+np.eye(h.shape[1]);self.detection=np.zeros((32,len(mapped)));self.pmean=np.zeros(len(mapped));self.support=np.zeros(len(mapped),bool);self.positive_mean=np.zeros(len(mapped))
        for start in range(0,len(mapped),256):
            sl=slice(start,min(start+256,len(mapped)));values=np.asarray(x[np.ix_(rows,columns[sl])],dtype=float);positive=values>0;count=positive.sum(0)
            self.support[sl]=count>=20;self.pmean[sl]=positive.mean(0);self.detection[:,sl]=np.linalg.solve(gram,h.T@positive/len(h));self.positive_mean[sl]=values.sum(0)/np.maximum(count,1)
        self.detection[:,~self.support]=0
        a=ad.read_h5ad(anchor_path,backed='r')
        try:
            az=[]
            with torch.no_grad():
                for start in range(0,a.n_obs,128):
                    values=a.X[start:start+128,mapped];values=values.toarray() if sparse.issparse(values) else np.asarray(values)
                    az.append(model.encode(torch.tensor(values,dtype=torch.float32))[0].numpy())
            ah=np.concatenate(az).astype(float)-self.zcenter;self.anchor_latent_mean=ah.mean(0);centered=ah-self.anchor_latent_mean;gram=centered.T@centered/len(ah)+np.eye(32);self.anchor_detection_delta=np.zeros_like(self.detection)
            for start in range(0,len(mapped),256):
                sl=slice(start,min(start+256,len(mapped)));values=a.X[:,mapped[sl]];values=values.toarray() if sparse.issparse(values) else np.asarray(values);positive=values>0;supported=(positive.sum(0)>=20)&self.support[sl]
                coef=np.linalg.solve(gram,centered.T@(positive-positive.mean(0))/len(ah));self.anchor_detection_delta[:,sl]=(coef-self.detection[:,sl])*supported
        finally:a.file.close()
        with torch.no_grad():
            self.decoded_anchor=[]
            for start in range(0,len(donors),64):self.decoded_anchor.append(model.decoder(model.encode(torch.tensor(donors[start:start+64,mapped]))[0]).numpy())
            self.decoded_anchor=np.concatenate(self.decoded_anchor)
        self.audit={'fit_cells':len(rows),'fit_max_stage':float(stages[rows].max()),'mapped_genes':len(mapped),'supported_genes':int(self.support.sum()),'method':'Fullmappedgene jointVAE/ODE with neural decoded log1p residual and .75 allanchor ridge detection; deterministic mean latent forecast','limitation':'Author adaptation; latestanchor transfer and conditionalpositive intercept approximate. No hidden future fitting.'}

    def detection_probability(self,h,sl):
        return np.clip(self.pmean[sl]+h@self.detection[:,sl]+.75*(h-self.anchor_latent_mean)@self.anchor_detection_delta[:,sl],1e-4,1-1e-4)

    def predict(self,target,*args,**kwargs):
        with torch.no_grad():
            predictions=[]
            for start in range(0,len(self.donors),64):
                z=self.model.encode(torch.tensor(self.donors[start:start+64,self.mapped]))[0]
                final=self.model.trajectory(z,torch.tensor([0.,target-self.cutoff]))[:,-1]
                predictions.append(self.model.decoder(final).numpy())
            self.decoded_future=np.concatenate(predictions)
        return super().predict(target,*args,**kwargs)

    def positive_log_change(self,h0,h1,sl,target):
        observed=self.donors[:,self.mapped[sl]].astype(float);delta=(self.decoded_future[:,sl]-self.decoded_anchor[:,sl])*self.support[sl]
        proposed=np.clip(observed+delta,1e-8,np.log1p(10000.));raw=np.expm1(observed)
        ratio=np.divide(np.expm1(proposed),raw,out=np.ones_like(raw),where=raw>0)
        return np.log(np.maximum(ratio,1e-12))

    def positive_log_value(self,h1,sl,target):
        value=self.positive_mean[sl]+self.decoded_future[:,sl]-self.decoded_anchor[:,sl]
        return np.log(np.maximum(np.expm1(np.clip(value,1e-8,np.log1p(10000.))),1e-8))
