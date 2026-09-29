"""Past-only nonlinear residual positive decoder over all exact mapped genes."""
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn
from feature_panel_forecast import FeaturePanelForecast


class PositiveResidualNet(nn.Module):
    def __init__(self,d,g):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(d,64),nn.Tanh(),nn.Linear(64,g))
        nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)
    def forward(self,h):return self.layers(h)


class NonlinearPositiveForecast(FeaturePanelForecast):
    def __init__(self,*args,residual_penalty=.1,steps=400,checkpoint=None,resume=False,emit=lambda **kw:None,**kwargs):
        if residual_penalty not in [.1,1.] or steps not in [0,2,4,400]:raise ValueError('Undeclared nonlinear experiment')
        super().__init__(*args,**kwargs)
        x,stages,cutoff=args[:3];panel,symbols=args[4:6]
        rows=np.flatnonzero(stages<=cutoff);counts=Counter(symbols)
        lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
        atlas=np.array([lookup[panel[i]] for i in self.mapped]);feat=[lookup[panel[i]] for i in self.features]
        with torch.no_grad():
            h=self.net.encode(torch.tensor((np.asarray(x[np.ix_(rows,feat)])-self.center)/self.scale))[0].numpy().astype(float)-self.zcenter
        torch.manual_seed(20260928)
        self.residual_net=PositiveResidualNet(h.shape[1],len(atlas))
        optimizer=torch.optim.Adam(self.residual_net.parameters(),lr=.001,weight_decay=1e-4)
        generator=torch.Generator().manual_seed(20260929)
        start=0;history=[];checkpoint=Path(checkpoint) if checkpoint else None
        if resume and checkpoint and checkpoint.exists():
            saved=torch.load(checkpoint,weights_only=False,map_location='cpu')
            if saved['residual_penalty']!=residual_penalty or saved['cutoff']!=cutoff or saved['fit_rows']!=len(rows):raise ValueError('Residual checkpoint mismatch')
            self.residual_net.load_state_dict(saved['net']);optimizer.load_state_dict(saved['optimizer'])
            generator.set_state(saved['sampling_rng']);start=saved['step'];history=saved['history']
            if start>steps:raise ValueError('Checkpoint beyond declared duration')
        support=torch.tensor(self.support,dtype=torch.float32)
        frequency=torch.tensor(np.maximum(self.pmean,.01),dtype=torch.float32)
        intercept=self.positive_mean-(self.positive_center*self.positive_coef).sum(0)
        for step in range(start,steps):
            ids=torch.randint(len(rows),(64,),generator=generator).numpy()
            raw=np.asarray(x[np.ix_(rows[ids],atlas)],float);mask=(raw>0).astype(np.float32)
            target=np.log(np.maximum(np.expm1(raw),1e-8))-(h[ids]@self.positive_coef+intercept)
            target=torch.tensor(target,dtype=torch.float32);mask=torch.tensor(mask)
            predicted=self.residual_net(torch.tensor(h[ids],dtype=torch.float32))*support
            conditional_loss=(((predicted-target)**2)*mask*support/frequency).mean()
            penalty=(predicted**2).mean();loss=conditional_loss+residual_penalty*penalty
            if not torch.isfinite(loss):raise ValueError('Nonfinite residual loss')
            optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.residual_net.parameters(),5.)
            if not torch.isfinite(norm):raise ValueError('Nonfinite residual gradient')
            optimizer.step()
            if (step+1)%50==0 or step+1==steps:
                record={'step':step+1,'loss':float(loss.detach()),'conditional_loss':float(conditional_loss.detach()),'residual_penalty_loss':float(penalty.detach()),'fit_max_stage':float(stages[rows].max())}
                history.append(record)
                if checkpoint:
                    torch.save({'net':self.residual_net.state_dict(),'optimizer':optimizer.state_dict(),'sampling_rng':generator.get_state(),'step':step+1,'residual_penalty':residual_penalty,'cutoff':cutoff,'fit_rows':len(rows),'history':history},checkpoint)
                emit(**record)
        self.residual_net.eval()
        self.audit.update(positive_decoder='8D latent to64tanh toallmappedgene residual',residual_penalty=residual_penalty,residual_training_steps=steps,residual_training_history=history,
                          scope='Own nonlinear conditional positive-log residual decoder, zero output initialization exactly retains baseline. All permitted past cells/mapped genes, inverse detection-frequency weighting capped at100, support20. No new latent dynamics/detection or future fitting; not full generative sampling.')

    def residual_value(self,h,sl):
        with torch.no_grad():
            hidden=self.residual_net.layers[1](self.residual_net.layers[0](torch.tensor(h,dtype=torch.float32)))
            final=self.residual_net.layers[-1]
            out=hidden@final.weight[sl].T+final.bias[sl]
        return out.numpy().astype(float)*self.support[sl]

    def positive_log_change(self,h0,h1,sl,target):
        return super().positive_log_change(h0,h1,sl,target)+self.residual_value(h1,sl)-self.residual_value(h0,sl)

    def positive_log_value(self,h1,sl,target):
        return super().positive_log_value(h1,sl,target)+self.residual_value(h1,sl)
