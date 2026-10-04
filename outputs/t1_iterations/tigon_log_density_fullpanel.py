"""Frozen log-density source-domain full-panel pilot; no retraining."""
import json
import traceback
from pathlib import Path
from collections import Counter
import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from cnf_manifold_flow import DensityFlowNet, train_manifold_density
from cnf_covariance_alignment import CovarianceAlignedTrajectory, symmetric_power
from full_anchor_slope_forecast import FullAnchorSlopeForecast
from scnode_past_fold_training import PastOnlyMatrix, past_features
from tigon_conditional_training import train_pair, forecast
from scnode_resource_preflight import peak_memory
from temporary_forecast_cache import TemporaryForecastCache
from offline_backtest import load_core, Panel
from run_t1 import digest
from iterate import now, append_event

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = HERE/'private/tigon_log_density_fullpanel_01'
PUBLIC = HERE/'TIGON_LOG_DENSITY_FULLPANEL_RESULTS.json'


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def panel_values(x, rows, mapped, columns, genes):
    result = np.zeros((len(rows), genes), dtype=np.float32)
    for start in range(0, len(mapped), 512):
        sl = slice(start, min(start+512,len(mapped)))
        result[:,mapped[sl]] = np.asarray(x[np.ix_(rows,columns[sl])],np.float32)
    return result


def main():
    if RUN.exists() or PUBLIC.exists():
        raise ValueError('Never duplicate or overwrite fresh past-fold run')
    RUN.mkdir(parents=True)
    events = RUN/'events.jsonl'
    emit = lambda kind, **kw: append_event(events,kind,**kw)
    cache = TemporaryForecastCache(HERE/'private/temporary_cache')
    report = {'status':'running','panels':[],'new_scoring_batch':True,
              'cutoff':7.75,'target':8.0,'original_readiness_gate_passed':False,'official_score':None}
    try:
        spec_path = HERE/'NEXT_TIGON_LOG_DENSITY_FULLPANEL.json'
        spec = json.loads(spec_path.read_text())
        names = spec['arms']
        cutoff,target = spec['cutoff'],spec['target']
        prepared = HERE/'private/associated_prepared_01'
        prep = json.loads((prepared/'report.json').read_text())
        source_files = ['tigon_log_density_fullpanel.py','tigon_conditional_training.py','tigon_conditional_core.py','scnode_past_fold_training.py','scnode_biological.py',
                        'scnode_joint_model.py','cnf_manifold_flow.py','cnf_density_flow.py',
                        'cnf_covariance_alignment.py','full_anchor_slope_forecast.py',
                        'partial_anchor_forecast.py','log1p_positive_forecast.py',
                        'feature_panel_forecast.py','ridge_conditional_head.py',
                        'offline_backtest.py','temporary_forecast_cache.py','scnode_resource_preflight.py','anchor_slope_calibration.py','neural_hurdle_forecast.py','robust_population.py']
        core,scorer_manifest = load_core()
        plan = {'created_utc':now(),'spec_sha256':digest(spec_path),
                'source_sha256':{f:digest(HERE/f) for f in source_files},
                'input_sha256':{},'cutoff':cutoff,'target':target,'candidate_names':names,
                'donor_seed':20260928,'scoring_seeds':[20260928,20260929,20260930],
                'joint_training_seed':20261002,'joint_steps':0,'cached_candidate_training_steps':200,'query_size':32,'query_noise_variance':.02,'KDE_variance':.1,'Adam_lr':.003,'weight_decay':.01,
                'new_arm_midpoint_max_step':.0625,'fresh_reference_steps':800,
                'scorer_manifest_sha256':digest(HERE/'private/scorer_source/manifest.json'),
                'population':spec['scope'],'submissions_allowed':0}
        for f,k in [('expression.npy','expression_sha256'),('genes.csv','genes_sha256'),
                    ('selected_metadata.csv','metadata_sha256')]:
            value=digest(prepared/f)
            if value!=prep[k]: raise ValueError('Prepared input changed: '+f)
            plan['input_sha256'][f]=value
        for filename in ['fresh_encoder.npz','fresh_reference.pt','generation.json']:
            plan.setdefault('frozen_artifact_sha256',{})['original/'+filename]=digest(HERE/'private/tigon_conditional_fullpanel_01'/filename)
        for name in names[2:]:
            plan['frozen_artifact_sha256'][name]=digest(HERE/'private/tigon_log_density_audit_01'/(name+'.pt'))
        panel_path=ROOT/'outputs/t1_run/T1__val.genes.txt' 
        plan['panel_sha256']=digest(panel_path)
        save(RUN/'plan.json',plan)
        emit('plan_frozen',sha256=digest(RUN/'plan.json'))
        panel=panel_path.read_text().splitlines()
        if len(panel)!=32285 or len(set(panel))!=len(panel): raise ValueError('Original panel mismatch')
        raw_x=np.load(prepared/'expression.npy',mmap_mode='r')
        stages=pd.read_csv(prepared/'selected_metadata.csv').numeric_stage.to_numpy(float)
        symbols=pd.read_csv(prepared/'genes.csv').symbol.fillna('').tolist()
        x=PastOnlyMatrix(raw_x,stages,cutoff)
        past=np.flatnonzero(stages<=cutoff)
        np.testing.assert_array_equal(np.unique(stages[past]),spec['fit_stages'])
        anchor_rows=np.flatnonzero(stages==cutoff)
        counts=Counter(symbols);lookup={g:i for i,g in enumerate(symbols) if g and counts[g]==1}
        mapped=np.array([i for i,g in enumerate(panel) if g in lookup],dtype=int)
        columns=np.array([lookup[panel[i]] for i in mapped],dtype=int)
        if len(mapped)!=26775: raise ValueError('Unique mapped panel changed')
        protected=np.setdiff1d(np.arange(len(panel)),mapped)
        for label,values in [('fit_rows',past),('mapped',mapped),('columns',columns),('anchor_rows',anchor_rows)]:
            np.save(RUN/(label+'.npy'),values)
        donor_rows=np.random.default_rng(20260928).choice(anchor_rows,1500,replace=False)
        np.testing.assert_array_equal(donor_rows,np.random.default_rng(20260928).choice(anchor_rows,1500,replace=False))
        np.save(RUN/'donor_rows.npy',donor_rows)
        donors=panel_values(x,donor_rows,mapped,columns,len(panel))
        anchor_path=RUN/'source_E7.75_anchor.h5ad'
        anchor=ad.AnnData(sparse.csr_matrix(panel_values(x,anchor_rows,mapped,columns,len(panel))),
                         obs=pd.DataFrame(index=[str(i) for i in anchor_rows]),
                         var=pd.DataFrame(index=panel))
        anchor.write_h5ad(anchor_path);del anchor
        emit('past_training_scope_frozen',fit_max_stage=float(stages[past].max()),fit_cells=len(past),
             anchor_cells=len(anchor_rows),mapped_genes=len(mapped),anchor_sha256=digest(anchor_path))
        atlas_features=past_features(x,past,columns,4096)
        atlas_guard=past_features(x,past,columns,384)
        official_lookup={g:i for i,g in enumerate(panel)}
        features=np.array([official_lookup[symbols[i]] for i in atlas_features])
        guard=np.array([official_lookup[symbols[i]] for i in atlas_guard])
        values=np.asarray(x[np.ix_(past,atlas_features)],np.float32)
        center=values.mean(0);scale=np.maximum(values.std(0),.1)
        normalized=(values-center)/scale
        fit=np.random.default_rng(20260928).choice(len(past),min(3000,len(past)),replace=False)
        pca=PCA(n_components=8,random_state=20260928).fit(normalized[fit])
        coordinates=pca.transform(normalized)
        whitening=np.maximum(coordinates.std(0),.1)
        basis=pca.components_/whitening[:,None];coordinates=coordinates/whitening
        np.savez_compressed(RUN/'fresh_encoder.npz',features=features,guard_features=guard,center=center,
                            scale=scale,basis=basis,pca_center=pca.mean_,whitening=whitening,pca_rows=past[fit])
        del values,normalized
        previous=HERE/'private/tigon_conditional_fullpanel_01'
        frozen_encoder=np.load(previous/'fresh_encoder.npz')
        for key,value in dict(features=features,guard_features=guard,center=center,scale=scale,basis=basis,pca_center=pca.mean_,whitening=whitening,pca_rows=past[fit]).items():
            np.testing.assert_array_equal(value,frozen_encoder[key])
        from tigon_conditional_core import ConditionalUOT
        flows={}
        for name,enabled in zip(names[2:],[False,True]):
            model=ConditionalUOT(growth=enabled)
            model.load_state_dict(torch.load(HERE/'private/tigon_log_density_audit_01'/(name+'.pt'),weights_only=True)['net'])
            flows[name]=model.eval()
        expected_controls=json.loads((previous/'generation.json').read_text())
        emit('frozen_encoder_and_candidate_models_loaded_no_training')
        torch.manual_seed(20260928)
        net=DensityFlowNet(basis,pca.mean_,cutoff,float(stages[past].min())-.25)
        checkpoint=torch.load(previous/'fresh_reference.pt',weights_only=True)
        net.load_state_dict(checkpoint['net']);history=checkpoint['history']
        net.eval()
        reference=FullAnchorSlopeForecast(x,stages,cutoff,donors,panel,symbols,net,center,scale,features,
                                          guard,anchor_path=anchor_path)
        reference.configure(.25,.75)
        with torch.no_grad():
            source_z=net.encode(torch.tensor((np.asarray(x[np.ix_(anchor_rows,atlas_features)],np.float32)-center)/scale))[0].numpy().astype(float)
        # Source and constructed anchor are the same population and rows: no domain transfer.
        challenge_z=source_z.copy()
        cs=np.cov(source_z,rowvar=False);cc=np.cov(challenge_z,rowvar=False)
        ridge=.05*(np.trace(cs)+np.trace(cc))/(2*cs.shape[0]);cs+=ridge*np.eye(8);cc+=ridge*np.eye(8)
        root=symmetric_power(cc,.5);inverse=symmetric_power(cc,-.5)
        full_map=inverse@symmetric_power(root@cs@root,.5)@inverse
        eig,vectors=np.linalg.eigh(full_map);full_map=(vectors*np.clip(eig,.5,2))@vectors.T
        transform=np.eye(8)+.25*(full_map-np.eye(8))
        np.testing.assert_allclose(transform,np.eye(8),atol=1e-6)
        np.savez_compressed(RUN/'fresh_alignment.npz',source_mean=source_z.mean(0),challenge_mean=challenge_z.mean(0),full_map=full_map,transform=transform)
        reference.net=CovarianceAlignedTrajectory(net,source_z.mean(0),challenge_z.mean(0),transform)
        reference.audit.update(training_history=history,fit_max_stage=cutoff,
                               method='Fresh cutoff production-family reference',
                               anchor_population='Same source E7.75 cohort; alignment effectively identity')
        generation={}
        for name in names:
            if name=='copy':
                pred,ids,audit=donors.copy(),np.arange(len(donors)),{'method':'persistence','fit_max_stage':cutoff}
            elif name=='fresh_pca8_cnf_incumbent_covariance_0.25':
                pred,ids,audit=reference.predict(target,'joint',1.,sampling='systematic')
            else:
                pred,ids,audit=forecast(reference,flows[name],donor_rows,target,RUN,name)
            np.testing.assert_array_equal(pred[:,protected],donors[ids][:,protected])
            if not np.isfinite(pred).all() or (pred<0).any():raise ValueError('Invalid forecast')
            if name in names[2:]:
                particles=np.load(RUN/(name+'_particles.npz'));logs=particles['logw']
                audit['log_weight_span']=float(logs.max()-logs.min())
                if audit['conditional_weight_ess']<750 or audit['log_weight_span']>np.log(4):raise ValueError('Predeclared forecast weight-stability gate failed: '+name)
            sha=cache.put(name,pred);del pred
            if name in names[:2] and sha!=expected_controls[name]['prediction_sha256']:
                raise ValueError('Frozen control differs from original pilot: '+name)
            if name=='fresh_pca8_cnf_incumbent_covariance_0.25':
                repeated,repeated_ids,_=reference.predict(target,'joint',1.,sampling='systematic')
            elif name=='copy':
                repeated,repeated_ids=donors.copy(),np.arange(len(donors))
            else:
                repeated,repeated_ids,_=forecast(reference,flows[name],donor_rows,target,RUN,name)
            if cache.put('_replay_'+name,repeated)!=sha:raise ValueError('Forecast replay differs: '+name)
            np.testing.assert_array_equal(ids,repeated_ids);del repeated
            emit('exact_forecast_replay_passed',candidate=name,prediction_sha256=sha)
            np.save(RUN/(name+'_indices.npy'),ids)
            generation[name]={'prediction_sha256':sha,'audit':audit}
            emit('forecast_frozen',candidate=name,prediction_sha256=sha)
        save(RUN/'generation.json',generation);report['generation']=generation
        emit('all_forecasts_frozen_before_target_expression',target=target)
        # This is the first permitted future expression access in this worker.
        available=np.flatnonzero(stages==target)
        for seed in plan['scoring_seeds']:
            rows=np.random.default_rng(seed).choice(available,2000,replace=False)
            np.save(RUN/f'target_rows_{seed}.npy',rows)
            future=panel_values(raw_x,rows,mapped,columns,len(panel))
            order=np.random.default_rng(seed).permutation(len(future));np.save(RUN/f'target_order_{seed}.npy',order)
            evaluator=Panel(core,future[order[:1000]],donors,seed)
            floor,ceiling=evaluator.metrics(donors),evaluator.metrics(future[order[1000:]])
            outcomes=[]
            for name in names:
                with cache.read(name,consume=False) as pred:raw=evaluator.metrics(pred)
                row={'candidate':name,'prediction_sha256':generation[name]['prediction_sha256'],
                     'raw_metrics':raw,**evaluator.aggregate(raw,floor,ceiling)}
                outcomes.append(row);emit('candidate_scored',seed=seed,**row)
            report['panels'].append({'seed':seed,'cutoff':cutoff,'target':target,'floor':floor,'ceiling':ceiling,'results':outcomes})
            save(RUN/'report.partial.json',report)
            del future,evaluator
        summary=[]
        for name in names:
            rows=[next(r for r in p['results'] if r['candidate']==name) for p in report['panels']]
            valid=all(r['calibration_valid'] for r in rows)
            summary.append({'candidate':name,'scores':[r['local_score'] for r in rows],
                            'mean_score':float(np.mean([r['local_score'] for r in rows])) if valid else None,
                            'raw_metrics':[r['raw_metrics'] for r in rows],'skills':[r['skills'] for r in rows],
                            'all_calibrations_valid':valid,
                            'mean_skills':{m:float(np.mean([r['skills'][m] for r in rows])) for m in rows[0]['skills']} if valid else None})
        by={r['candidate']:r for r in summary}
        controls=[by[n] for n in names[:2]];passing=[]
        for name in names[2:]:
            new=by[name]
            passed=new['all_calibrations_valid'] and all(c['all_calibrations_valid'] for c in controls) and all(
                new['scores'][i]>max(c['scores'][i] for c in controls) for i in range(3)) and all(
                new['mean_skills'][m]>=max(c['mean_skills'][m] for c in controls) for m in new['mean_skills'])
            if passed:passing.append(name)
        contrast=None
        if all(by[n]['all_calibrations_valid'] for n in names[2:]):
            contrast={m:by[names[3]]['mean_skills'][m]-by[names[2]]['mean_skills'][m] for m in by[names[3]]['mean_skills']}
        report.update(status='completed',completed_utc=now(),summary=summary,
                      passing_candidates=passing, growth_enabled_minus_disabled_mean_skills=contrast,
                      expand_to_16_resamples=any(by[n]['mean_score']>=60 for n in passing),
                      scope=spec['scope'],plan_sha256=digest(RUN/'plan.json'),
                      resource_peak_process_working_set_bytes=peak_memory())
    except Exception as exc:
        (RUN/'traceback.txt').write_text(traceback.format_exc())
        report.update(status='failed',error=type(exc).__name__+': '+str(exc),failed_utc=now(),passing_candidates=[])
    finally:
        cache.close()
    report['resource_peak_process_working_set_bytes']=peak_memory()
    if peak_memory()>=16*1024**3:report.update(status='failed',error='16GiB resource gate failed',passing_candidates=[])
    save(RUN/'report.json',report)
    report['report_sha256']=digest(RUN/'report.json');save(PUBLIC,report)
    emit('batch_finished',status=report['status'],report_sha256=report['report_sha256'])
    if report.get('panels'):
        from index_scores import main as index_scores
        index_scores()
    print(json.dumps({'status':report['status'],'report':str(PUBLIC)}))


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
