"""One log-density loss-family past audit; no future data or benchmark scores."""
import copy
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from sklearn.decomposition import PCA
from tigon_conditional_core import ConditionalUOT, integrate, log_density
from tigon_objective_signal import components, ConstantField
from scnode_past_fold_training import PastOnlyMatrix
from scnode_resource_preflight import peak_memory
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'t1_run'))
from run_t1 import digest
from iterate import append_event, now

HERE=Path(__file__).resolve().parent
RUN=HERE/'private/tigon_log_density_audit_01'
PUBLIC=HERE/'TIGON_LOG_DENSITY_AUDIT_RESULTS.json'


def draw(groups, generator):
    query=groups[1][torch.randperm(len(groups[1]),generator=generator)[:32]]+torch.randn(32,8,generator=generator)*np.sqrt(.02)
    start=groups[0][torch.randperm(len(groups[0]),generator=generator)[:32]]+torch.randn(32,8,generator=generator)*np.sqrt(.02)
    return query, start


def losses(model, groups, query, start):
    target=log_density(query,groups[1]);density=query.new_zeros(())
    for _ in [0,0]:
        z,_,delta,_=integrate(model,query.clone(),.25,0.)
        predicted=log_density(z,groups[0])-delta[:,0]
        density=density+((target-predicted)/query.shape[1]).square().mean()
    _,_,_,energy=integrate(model,start.clone(),0.,.25)
    return density,.25*energy.mean()


def main():
    if RUN.exists() or PUBLIC.exists():raise ValueError('No duplicate gradient-balance audit')
    RUN.mkdir(parents=True);started=time.perf_counter()
    emit=lambda event,**kw:append_event(RUN/'events.jsonl',event,**kw)
    report={'status':'running','new_scoring_batch':False,'raw_metrics':None,'skills':None,
            'reward_delta':0,'passing_candidates':[],'original_readiness_gate_passed':False}
    try:
        source=HERE/'private/tigon_conditional_fullpanel_01';prepared=HERE/'private/associated_prepared_01'
        spec=json.loads((HERE/'NEXT_TIGON_LOG_DENSITY_AUDIT.json').read_text())
        diagnostic=json.loads((HERE/'TIGON_OBJECTIVE_SIGNAL_RESULTS.json').read_text())
        initial_result=diagnostic['diagnostics']['shared_initial_disabled']
        alpha=1.  # New fixed loss units; no gradient balance or multiplier search.
        if not np.isfinite(alpha) or not 0<alpha<1000:raise ValueError('Invalid fixed past ratio')
        plan={'created_utc':now(),'spec_sha256':digest(HERE/'NEXT_TIGON_LOG_DENSITY_AUDIT.json'),
              'alpha':alpha,'diagnostic_sha256':digest(HERE/'TIGON_OBJECTIVE_SIGNAL_RESULTS.json'),
              'source_sha256':{f:digest(HERE/f) for f in ['tigon_log_density_audit.py','tigon_objective_signal.py','tigon_conditional_core.py','scnode_past_fold_training.py','scnode_resource_preflight.py']},
              'pilot_artifact_sha256':{f:digest(source/f) for f in ['plan.json','fresh_encoder.npz','fit_rows.npy','tigon_shared_initial.pt','conditional_tigon_growth_disabled.pt','conditional_tigon_growth_enabled.pt']},
              'input_sha256':{f:digest(prepared/f) for f in ['expression.npy','selected_metadata.csv','genes.csv']}}
        original_plan=json.loads((source/'plan.json').read_text())
        if plan['input_sha256']!=original_plan['input_sha256']:raise ValueError('Original inputs changed')
        (RUN/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');emit('fixed_gradient_ratio_plan',alpha=alpha,plan_sha256=digest(RUN/'plan.json'))
        enc=np.load(source/'fresh_encoder.npz');symbols=pd.read_csv(prepared/'genes.csv').symbol.fillna('').tolist()
        panel=(HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines();lookup={g:i for i,g in enumerate(symbols)}
        columns=np.array([lookup[panel[i]] for i in enc['features']]);stages=pd.read_csv(prepared/'selected_metadata.csv').numeric_stage.to_numpy(float)
        x=PastOnlyMatrix(np.load(prepared/'expression.npy',mmap_mode='r'),stages,7.75);rows=np.load(source/'fit_rows.npy')
        if np.unique(stages[rows]).tolist()!=[7.5,7.75]:raise ValueError('Past scope changed')
        normalized=(np.asarray(x[np.ix_(rows,columns)],np.float32)-enc['center'])/enc['scale']
        pca_rows=np.searchsorted(rows,enc['pca_rows'])
        np.testing.assert_array_equal(rows[pca_rows],enc['pca_rows'])
        pca=PCA(n_components=8,random_state=20260928).fit(normalized[pca_rows])
        coordinates=pca.transform(normalized)
        whitening=np.maximum(coordinates.std(0),.1)
        np.testing.assert_array_equal(pca.components_/whitening[:,None],enc['basis'])
        np.testing.assert_array_equal(pca.mean_,enc['pca_center'])
        z=torch.tensor(coordinates/whitening,dtype=torch.float32);del normalized,coordinates
        emit('exact_original_encoder_reconstruction_passed')
        groups=[z[stages[rows]==t] for t in [7.5,7.75]]
        initial=torch.load(source/'tigon_shared_initial.pt',weights_only=True)
        models={};histories={};stream_hashes={}
        expected_stream=next(json.loads(l)['draw_sha256'] for l in (source/'events.jsonl').read_text().splitlines() if json.loads(l)['event']=='matched_tigon_stream')
        for loss_kind in ['log_density']:
            for enabled in [False,True]:
                suffix='enabled' if enabled else 'disabled';name=loss_kind+'_'+suffix
                model=ConditionalUOT(growth=enabled);model.load_state_dict(initial)
                optimizer=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=.01)
                generator=torch.Generator().manual_seed(20261002);stream=hashlib.sha256();history=[]
                for step in range(200):
                    query,start=draw(groups,generator)
                    for value in [query,start]:stream.update(value.numpy().tobytes())
                    density,energy=losses(model,groups,query,start)
                    loss=density+energy
                    if not torch.isfinite(loss):raise ValueError('Nonfinite balanced objective')
                    optimizer.zero_grad();loss.backward()
                    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite balanced gradient')
                    if peak_memory()>=16*1024**3:raise ValueError('16GiB resource gate failed')
                    optimizer.step()
                    if (step+1)%25==0:
                        record={'candidate':name,'step':step+1,'optimized_loss':float(loss.detach()),'original_density':float(density.detach()),'original_energy':float(energy.detach())}
                        history.append(record);emit('training_checkpoint',**record)
                stream_hashes[name]=stream.hexdigest()
                if stream.hexdigest()!=expected_stream:raise ValueError('Original paired training stream changed')
                torch.save({'net':model.state_dict(),'optimizer':optimizer.state_dict(),'generator':generator.get_state(),'history':history,'alpha':alpha,'seed':20261002,'steps':200,'fit_max_stage':7.75,'loss_kind':loss_kind},RUN/(name+'.pt'))
                models[name]=copy.deepcopy(model).eval();histories[name]=history
        for enabled in [False,True]:
            suffix='enabled' if enabled else 'disabled'
            model=ConditionalUOT(growth=enabled);model.load_state_dict(initial);models['initial_'+suffix]=model
            model=ConditionalUOT(growth=enabled);model.load_state_dict(torch.load(source/('conditional_tigon_growth_'+suffix+'.pt'),weights_only=True)['net']);models['cached_original_'+suffix]=model
        models['zero_field']=ConstantField(torch.zeros(8));models['past_centroid_translation']=ConstantField((groups[1].mean(0)-groups[0].mean(0))/.25);panels=[]
        for seed in spec['diagnostic_seeds']:
            query,start=draw(groups,torch.Generator().manual_seed(seed));results={}
            np.savez_compressed(RUN/('query_'+str(seed)+'.npz'),query=query.numpy(),start=start.numpy())
            for name,model in models.items():
                result=components(model,groups,query.clone(),start.clone())
                log_loss,_=losses(model,groups,query.clone(),start.clone())
                result['log_density_MSE_per_dimension']=float(log_loss.detach())
                final,logs,_,_=integrate(model,start.clone(),0.,.25)
                weights=torch.softmax(logs.detach().flatten(),0)
                result.update(displacement_RMS=float((final.detach()-start).square().mean().sqrt()),ESS=float(1/weights.square().sum()),log_weight_span=float(logs.detach().max()-logs.detach().min()))
                results[name]=result
            panels.append({'seed':seed,'results':results});emit('past_diagnostic_seed_completed',seed=seed)
        passing=[];failures={}
        for suffix in ['disabled','enabled']:
            name='log_density_'+suffix;reasons=[]
            for panel_record in panels:
                results=panel_record['results'];candidate=results[name];zero=results['zero_field']
                if candidate['log_density_MSE_per_dimension']>.9*min(zero['log_density_MSE_per_dimension'],results['cached_original_'+suffix]['log_density_MSE_per_dimension']):reasons.append('log_density_reduction_under_10percent')
                if candidate['moving_energy_weighted']>results['past_centroid_translation']['moving_energy_weighted']:reasons.append('energy_above_past_centroid')
                if not all(np.isfinite(candidate[k]) for k in ['objective','log_density_MSE_per_dimension','ESS','log_weight_span']) or candidate['ESS']<16 or candidate['log_weight_span']>np.log(4):reasons.append('nonfinite_or_weight_concentration')
            if not reasons:passing.append(name)
            failures[name]=sorted(set(reasons))
        report.update(status='completed',completed_utc=now(),alpha=alpha,training_stream_sha256=stream_hashes,
                      training_history=histories,past_diagnostic_panels=panels,passing_candidates=passing,failure_reasons=failures,
                      retire_fixed_log_density_path=not bool(passing),plan_sha256=digest(RUN/'plan.json'),
                      scope='Past-only repeated source development; new noise queries, same KDE centers, not held-out cells or embryos. No target expression, forecast, benchmark score or reward. No automatic full-panel retry.')
    except Exception as exc:
        (RUN/'traceback.txt').write_text(traceback.format_exc());report.update(status='failed',error=type(exc).__name__+': '+str(exc))
    report.update(elapsed_seconds=time.perf_counter()-started,peak_process_working_set_bytes=peak_memory())
    (RUN/'report.json').write_text(json.dumps(report,indent=2)+'\n');report['report_sha256']=digest(RUN/'report.json');PUBLIC.write_text(json.dumps(report,indent=2)+'\n')
    emit('batch_finished',status=report['status']);print(json.dumps({'status':report['status'],'report':str(PUBLIC)}))


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
