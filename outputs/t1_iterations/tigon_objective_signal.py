"""Predeclared past-only loss/gradient diagnostic; no training or new scores."""
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from tigon_conditional_core import ConditionalUOT, integrate, log_density, objective, conditional_weights
from scnode_past_fold_training import PastOnlyMatrix
from scnode_resource_preflight import peak_memory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'t1_run'))
from run_t1 import digest
from iterate import append_event, now

HERE = Path(__file__).resolve().parent
RUN = HERE/'private/tigon_objective_signal_01'
PUBLIC = HERE/'TIGON_OBJECTIVE_SIGNAL_RESULTS.json'


class ConstantField(ConditionalUOT):
    def __init__(self, velocity):
        super().__init__(growth=False)
        self.constant = torch.nn.Parameter(velocity.clone())
    def fields(self, t, z):
        # z*0 preserves an exact zero divergence autograd path.
        return z*0+self.constant, z[:, :1]*0


def components(model, groups, query, start):
    density = query.new_zeros(())
    target = log_density(query, groups[1]).exp()
    predicted = None
    for _ in [0, 0]:  # Same origin + adjacent terms as the fixed two-stage pilot.
        z, _, delta, _ = integrate(model, query.clone(), .25, 0.)
        predicted = (log_density(z, groups[0])-delta[:, 0]).exp()
        density = density + 1e4*(target-predicted).square().mean()
    _, _, _, energy = integrate(model, start.clone(), 0., .25)
    energy = .25*energy.mean()
    total = density+energy
    expected = objective(model, groups, [0., .25], [query.clone()], start.clone())
    torch.testing.assert_close(total, expected)
    params = list(model.named_parameters())
    vectors = []
    for term in [density, energy]:
        grads = torch.autograd.grad(term, [v for _, v in params], retain_graph=True, allow_unused=True)
        vectors.append(torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for (_, p), g in zip(params, grads)]))
    a, b = vectors
    norm_a, norm_b = float(a.norm()), float(b.norm())
    cosine = float(torch.dot(a, b)/(a.norm()*b.norm())) if norm_a and norm_b else None
    return {'density_MSE_weighted_origin_plus_adjacent':float(density.detach()),
            'moving_energy_weighted':float(energy.detach()), 'objective':float(total.detach()),
            'density_gradient_norm':norm_a, 'energy_gradient_norm':norm_b,
            'gradient_cosine':cosine, 'combined_gradient_norm':float((a+b).norm()),
            'target_KDE_density_min':float(target.detach().min()), 'target_KDE_density_max':float(target.detach().max()),
            'predicted_KDE_density_min':float(predicted.detach().min()), 'predicted_KDE_density_max':float(predicted.detach().max())}


def main():
    if RUN.exists() or PUBLIC.exists(): raise ValueError('Never duplicate diagnostic')
    RUN.mkdir(parents=True); started=time.perf_counter()
    emit=lambda event, **kw: append_event(RUN/'events.jsonl', event, **kw)
    report={'status':'running','new_scoring_batch':False,'raw_metrics':None,'skills':None,
            'reward_delta':0,'passing_candidates':[],'original_readiness_gate_passed':False}
    try:
        source=HERE/'private/tigon_conditional_fullpanel_01'
        prepared=HERE/'private/associated_prepared_01'
        plan={'created_utc':now(),'spec_sha256':digest(HERE/'NEXT_TIGON_OBJECTIVE_SIGNAL.json'),
              'source_sha256':{f:digest(HERE/f) for f in ['tigon_objective_signal.py','tigon_conditional_core.py','scnode_past_fold_training.py','scnode_resource_preflight.py']},
              'pilot_artifact_sha256':{f:digest(source/f) for f in ['plan.json','fresh_encoder.npz','tigon_shared_initial.pt','conditional_tigon_growth_disabled.pt','conditional_tigon_growth_enabled.pt']},
              'input_sha256':{f:digest(prepared/f) for f in ['expression.npy','selected_metadata.csv','genes.csv']}}
        (RUN/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');emit('diagnostic_plan_frozen',sha256=digest(RUN/'plan.json'))
        encoder=np.load(source/'fresh_encoder.npz')
        symbols=pd.read_csv(prepared/'genes.csv').symbol.fillna('').tolist()
        panel=(HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
        lookup={g:i for i,g in enumerate(symbols)}
        columns=np.array([lookup[panel[i]] for i in encoder['features']])
        stages=pd.read_csv(prepared/'selected_metadata.csv').numeric_stage.to_numpy(float)
        raw=np.load(prepared/'expression.npy',mmap_mode='r'); x=PastOnlyMatrix(raw, stages, 7.75)
        rows=np.load(source/'fit_rows.npy')
        if np.unique(stages[rows]).tolist()!=[7.5,7.75]:raise ValueError('Past scope changed')
        values=torch.tensor((np.asarray(x[np.ix_(rows,columns)],np.float32)-encoder['center'])/encoder['scale'])
        z=(values-torch.tensor(encoder['pca_center']))@torch.tensor(encoder['basis']).T
        del values
        groups=[z[stages[rows]==t] for t in [7.5,7.75]]
        generator=torch.Generator().manual_seed(20261002)
        query=groups[1][torch.randperm(len(groups[1]),generator=generator)[:32]]+torch.randn(32,8,generator=generator)*np.sqrt(.02)
        start=groups[0][torch.randperm(len(groups[0]),generator=generator)[:32]]+torch.randn(32,8,generator=generator)*np.sqrt(.02)
        np.savez_compressed(RUN/'fixed_queries.npz',query=query.numpy(),start=start.numpy())
        initial=torch.load(source/'tigon_shared_initial.pt',weights_only=True)
        models={}
        for enabled in [False,True]:
            suffix='enabled' if enabled else 'disabled'
            model=ConditionalUOT(growth=enabled);model.load_state_dict(initial);models['shared_initial_'+suffix]=model
            model=ConditionalUOT(growth=enabled)
            model.load_state_dict(torch.load(source/('conditional_tigon_growth_'+suffix+'.pt'),weights_only=True)['net'])
            models['trained_'+suffix]=model
        centroid_velocity=(groups[1].mean(0)-groups[0].mean(0))/.25
        models['zero_field']=ConstantField(torch.zeros(8))
        models['past_centroid_translation']=ConstantField(centroid_velocity)
        report['diagnostics']={}
        for name,model in models.items():
            result=components(model,groups,query.clone(),start.clone())
            final,logs,_,_=integrate(model,start.clone(),0.,.25)
            weights,ess=conditional_weights(logs.detach())
            result.update(past_displacement_RMS=float((final.detach()-start).square().mean().sqrt()),
                          normalized_weight_ESS=float(ess),log_weight_range=[float(logs.detach().min()),float(logs.detach().max())])
            if not all(np.isfinite(v) for v in [result['objective'],result['density_gradient_norm'],result['energy_gradient_norm']]):raise ValueError('Nonfinite diagnostic')
            if peak_memory()>=16*1024**3:raise ValueError('16GiB diagnostic resource gate failed')
            report['diagnostics'][name]=result;emit('objective_signal_measured',candidate=name,**result)
        report.update(status='completed',completed_utc=now(),plan_sha256=digest(RUN/'plan.json'),
                      scope='Stored past encoder/models only; 7.5/7.75 expression. Fixed original first query stream. No retraining, future expression, benchmark scores or validation claim.',
                      group_sizes=[len(g) for g in groups],past_centroid_velocity_norm=float(centroid_velocity.norm()))
    except Exception as exc:
        (RUN/'traceback.txt').write_text(traceback.format_exc());report.update(status='failed',error=type(exc).__name__+': '+str(exc))
    report.update(elapsed_seconds=time.perf_counter()-started,peak_process_working_set_bytes=peak_memory())
    (RUN/'report.json').write_text(json.dumps(report,indent=2)+'\n');report['report_sha256']=digest(RUN/'report.json')
    PUBLIC.write_text(json.dumps(report,indent=2)+'\n');emit('batch_finished',status=report['status'])
    print(json.dumps({'status':report['status'],'report':str(PUBLIC)}))


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
