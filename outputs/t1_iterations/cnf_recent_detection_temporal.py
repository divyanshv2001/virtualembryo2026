"""Learner latent-dimensionality ablations with fixed stationary controls."""
import json
import argparse
import importlib.metadata
from collections import Counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import torch
from cnf_manifold_flow import DensityFlowNet,train_manifold_density
from feature_panel_forecast import FeaturePanelForecast
from recent_detection_head import RecentDetectionForecast
from past_encoder_panel import fit_past_encoder
from train_extended_atlas import HERE
from offline_backtest import load_core, Panel
from run_t1 import digest
from iterate import now, append_event


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    root=HERE.parents[1];data=HERE/'private/associated_prepared_01'
    out=HERE/'private/cnf_recent_detection_temporal_01'
    if out.exists() and not args.resume:raise ValueError('Preserve previous run; use --resume')
    prepared=json.loads((data/'report.json').read_text())
    for f,k in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(data/f)!=prepared[k]:raise ValueError('Prepared data changed')
    plan={'created_before_training_utc':now(),'folds':[[8.,9.],[8.25,9.25]],
        'configs':['copy','unit16','saved384_d0.0','dimension_neutral_d8','dimension_growth1_d8','flow_abundance_s1.0','flow_joint_systematic_s1.0','recent1_joint','recent2_joint'],
        'donor_count':1500,'truth_count':1000,'ceiling_count':1000,'seed':20260928,
        'prepared_report_sha256':digest(data/'report.json'),
        'source_sha256':{f:digest(HERE/f) for f in ['cnf_recent_detection_temporal.py','detection_transfer.py',
            'challenge_transfer.py','robust_population.py','constrained_forecast.py','transfer_genes.py','offline_backtest.py','annotation_trend.py','neural_ode_forecast.py','neural_hurdle_forecast.py','transport_latent_flow.py','feature_panel_forecast.py','recent_detection_head.py','windowed_positive_head.py','past_encoder_panel.py','ridge_conditional_head.py','cnf_density_flow.py','cnf_manifold_flow.py','finish_horizon_batch.py','summarize_horizon_batch.py','index_scores.py']},
        'fit':'At each cutoff, past-fitted archived8D PCA whitened with current past cells. Train3x64 tanh CNF with400/800-mandated configuration:800 updates, energy.1, manifold10, batch64, RK4.125 and midpoint.0625. Full conditional positive ridge1 heads use permitted past cells only.',
        'primary_objective':'Unchanged joint full-panel score. Original >72 Monte Carlo/temporal readiness gate and all metric mean skills>=50 remain required.',
        'ablation':'Transport-to-density learner change already chosen from exposed challenge development; this is rolling consistency assessment of fixed800-update configuration, not new blind validation. Two abundance strengths.5/1; five unchanged archived controls; scoring PCA/metrics unchanged.',
        'scope':'Two frozen800-update flow strengths, two reused development folds,14 evaluations including five matched controls. Sampled prepared cohort and384 features, no whole raw-atlas training or independent challenge validation. Missing official genes remain placeholders. No positive result implies extrapolation certification.',
        'gate':'No official upload. Retain every metric, invalid calibration and convergence failure. No evaluation-seed search.',
        'literature':'CNF_MANIFOLD_REFERENCE.json documents author likelihood/energy/density review. This is our CPU adaptation assessed on reused temporal development folds; not a full published reproduction or extrapolation guarantee.',
        'submissions_allowed':0,'jev_requests_allowed':0}
    old=HERE/'private/dimensional_transport_horizon_01'
    feature_root=HERE/'private/neural_ode_horizon_01'
    plan['archived_control_report_sha256']=digest(old/'report.json')
    plan['archived_control_predictions_sha256']={f'cutoff_{c}/{n}.npy':digest(old/f'cutoff_{c}'/(n+'.npy')) for c,_ in plan['folds'] for n in ['copy','unit16','saved384_d0.0','dimension_neutral_d8','dimension_growth1_d8']}
    proxy_root=HERE/'private/growth_prior_audit_repair_01'
    plan['growth_proxy_sha256']={str(c):digest(proxy_root/f'scores_{c}.npz') for c,_ in plan['folds']}
    plan['feature_archive_sha256']={str(c):digest(feature_root/f'cutoff_{c}'/'neural_b0.0.npz') for c,_ in plan['folds']}
    plan['author_reference_sha256']=digest(HERE/'CNF_MANIFOLD_REFERENCE.json')
    plan['past_encoder_sha256']={str(c):digest(old/f'cutoff_{c}/neutral_d8_flow.npz') for c,_ in plan['folds']}
    archive=HERE/'private/cnf_hurdle_temporal_01'
    plan['fit']='Reuse exact archived4096-gene8D encoder and800-update flow for each cutoff8/8.25. Recompute unchanged global abundance heads and verify archive coefficients/predictions exactly. Fit detection windows1/2 actual permitted past stages. No dynamics/encoder training.'
    plan['ablation']='Only recent detection head changes; global abundance and previously frozen dynamics fixed. Joint systematic strength1, windows1/2 versus global detection, full-panel/calibration unchanged.'
    plan['scope']='Two reused source-cohort one-day folds, seven archived controls plus recent1/2 joint forecasts:18 scores. Missing official genes remain zero placeholders; no blind or independent embryo validation.'
    plan['flow_archive_sha256']={str(c):{f:digest(archive/f'cutoff_{c}'/f) for f in ['encoder.npz','training.pt','heads.npz']} for c,_ in plan['folds']}
    plan['flow_control_sha256']={f'cutoff_{c}/{n}.npy':digest(archive/f'cutoff_{c}'/(n+'.npy')) for c,_ in plan['folds'] for n in ['flow_abundance_s1.0','flow_joint_systematic_s1.0']}
    plan['dependencies']={k:importlib.metadata.version(k) for k in ['torch','torchdiffeq','geomloss','numpy','scipy','scikit-learn']}
    if args.resume:
        existing=json.loads((out/'plan.json').read_text())
        for key in ['source_sha256','prepared_report_sha256','author_reference_sha256','dependencies','configs','growth_proxy_sha256','feature_archive_sha256','past_encoder_sha256','flow_archive_sha256','flow_control_sha256']:
            if existing[key]!=plan[key]:raise ValueError('Resume plan mismatch: '+key)
        plan=existing
    else:
        out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2))
    events=out/'events.jsonl'
    if not args.resume:
        for f in plan['source_sha256']:(out/f).write_bytes((HERE/f).read_bytes())
    append_event(events,'neural_plan_resumed' if args.resume else 'matched_horizon_plan_frozen',sha256=digest(out/'plan.json'))
    x=np.load(data/'expression.npy',mmap_mode='r');metadata=pd.read_csv(data/'selected_metadata.csv');stages=metadata.numeric_stage.to_numpy(float)
    symbols=pd.read_csv(data/'genes.csv').symbol.fillna('').tolist()
    panel=(root/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
    counts=Counter(symbols);lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
    official=np.array([i for i,s in enumerate(panel) if s in lookup]);atlas=np.array([lookup[panel[i]] for i in official])
    def values(rows):
        result=np.zeros((len(rows),len(panel)),dtype=np.float32)
        result[:,official]=np.asarray(x[np.ix_(rows,atlas)])
        return result
    core,manifest=load_core();report={'plan':plan,'folds':[],'official_score':None,'local_gate_passed':False,
        'scorer_manifest_sha256':digest(HERE/'private/scorer_source/manifest.json')}
    if args.resume and (out/'report.partial.json').exists():report=json.loads((out/'report.partial.json').read_text());report.pop('in_progress_fold',None)
    for cutoff,target in plan['folds']:
        if any(f['cutoff']==cutoff for f in report['folds']):continue
        folder=out/f'cutoff_{cutoff}';folder.mkdir(exist_ok=args.resume)
        donor_rows=np.sort(np.random.default_rng(plan['seed']).choice(np.flatnonzero(stages==cutoff),1500,replace=False))
        donors=values(donor_rows);np.save(folder/'donor_rows.npy',donor_rows)
        old_model=feature_root/f'cutoff_{cutoff}'
        if digest(old_model/'neural_b0.0.npz')!=plan['feature_archive_sha256'][str(cutoff)]:raise ValueError('Feature archive changed')
        saved=np.load(old_model/'neural_b0.0.npz');features=saved['features']
        proxy_path=proxy_root/f'scores_{cutoff}.npz'
        if digest(proxy_path)!=plan['growth_proxy_sha256'][str(cutoff)]:raise ValueError('Growth inputs changed')
        proxy=np.load(proxy_path)
        encoder_path=old/f'cutoff_{cutoff}/neutral_d8_flow.npz'
        if digest(encoder_path)!=plan['past_encoder_sha256'][str(cutoff)]:raise ValueError('Past encoder changed')
        archived_folder=archive/f'cutoff_{cutoff}'
        for name,sha in plan['flow_archive_sha256'][str(cutoff)].items():
            if digest(archived_folder/name)!=sha:raise ValueError('Archived encoder/flow changed')
        encoded=np.load(archived_folder/'encoder.npz');guard_features=encoded['guard_features'];features=encoded['features'];center=encoded['center'];scale=encoded['scale'];basis=encoded['basis']
        net=DensityFlowNet(basis,encoded['pca_center'],cutoff,7.25)
        saved_flow=torch.load(archived_folder/'training.pt',weights_only=False,map_location='cpu');net.load_state_dict(saved_flow['net']);net.eval();history=saved_flow['history']
        program=FeaturePanelForecast(x,stages,cutoff,donors,panel,symbols,net,center,scale,features,guard_features)
        saved_heads=np.load(archived_folder/'heads.npz')
        for key in ['positive_coef','positive_center','positive_mean','zcenter','support']:np.testing.assert_array_equal(getattr(program,key),saved_heads[key])
        np.testing.assert_array_equal(program.predict(target,'abundance',1.)[0],np.load(archived_folder/'flow_abundance_s1.0.npy'))
        append_event(events,'archived_full_panel_abundance_verified_exactly',cutoff=cutoff)
        program.audit.update(method='800-update likelihood/energy/manifold flow temporal check',training_steps=800,energy_weight=.1,density_weight=10.,training_history=history,scope='Past-only fit at each rolling cutoff; source-cohort diagnostic with zero placeholders for absent challenge genes. Reused developmental folds, not independent embryos or blind validation.')
        np.savez_compressed(folder/'heads.npz',positive_coef=program.positive_coef,positive_center=program.positive_center,positive_mean=program.positive_mean,zcenter=program.zcenter,support=program.support)
        oldfolder=old/f'cutoff_{cutoff}'
        np.testing.assert_array_equal(donor_rows,np.load(oldfolder/'donor_rows.npy'))
        oldgeneration=json.loads((oldfolder/'generation.json').read_text())
        generation={}
        for name in plan['configs']:
            if name.startswith('recent'):
                window=int(name[len('recent')]);recent=RecentDetectionForecast(x,stages,cutoff,donors,panel,symbols,net,center,scale,features,guard_features,head_stages=window)
                np.testing.assert_array_equal(recent.positive_coef,saved_heads['positive_coef'])
                pred,indices,audit=recent.predict(target,'joint',1.,sampling='systematic')
            else:
                previous=name;source=(archived_folder if name.startswith('flow_') else oldfolder)/(previous+'.npy')
                sha=(plan['flow_control_sha256'] if name.startswith('flow_') else plan['archived_control_predictions_sha256'])[f'cutoff_{cutoff}/{previous}.npy']
                if digest(source)!=sha:raise ValueError('Archived prediction changed')
                pred=np.load(source);indices=np.load(source.with_name(previous+'_indices.npy'))
                generation_source=json.loads((source.parent/'generation.json').read_text())
                audit={**generation_source[previous]['audit'],'replayed_from':str(source.relative_to(HERE))}
            np.save(folder/(name+'.npy'),pred);np.save(folder/(name+'_indices.npy'),indices)
            generation[name]={'audit':audit,'prediction_sha256':digest(folder/(name+'.npy'))}
            append_event(events,'matched_horizon_forecast_generated',cutoff=cutoff,candidate=name)
            del pred
        (folder/'generation.json').write_text(json.dumps(generation,indent=2))
        append_event(events,'all_fold_predictions_frozen_before_target_read',cutoff=cutoff,target=target)
        target_rows=np.sort(np.random.default_rng(plan['seed']).choice(np.flatnonzero(stages==target),2000,replace=False))
        future=values(target_rows);order=np.random.default_rng(plan['seed']).permutation(2000)
        evaluator=Panel(core,future[order[:1000]],donors,plan['seed'])
        floor=evaluator.metrics(donors);ceiling=evaluator.metrics(future[order[1000:]])
        oldreport=json.loads((old/'report.json').read_text())
        oldfold=next(f for f in oldreport['folds'] if f['cutoff']==cutoff)
        if floor!=oldfold['floor'] or ceiling!=oldfold['ceiling']:raise ValueError('Calibration panels differ')
        results=[]
        for name in plan['configs']:
            pred=np.load(folder/(name+'.npy'),mmap_mode='r');raw=evaluator.metrics(pred)
            result={'candidate':name,'raw_metrics':raw,**evaluator.aggregate(raw,floor,ceiling)}
            results.append(result);append_event(events,'matched_horizon_candidate_scored',cutoff=cutoff,
                candidate=name,score=result['local_score'],valid=result['calibration_valid'],raw_metrics=result['raw_metrics'],skills=result['skills'])
            (out/'report.partial.json').write_text(json.dumps({**report,'in_progress_fold':{'cutoff':cutoff,'target':target,'floor':floor,'ceiling':ceiling,'results':results,'generation':generation}},indent=2))
            del pred
        report['folds'].append({'cutoff':cutoff,'target':target,'floor':floor,'ceiling':ceiling,
            'results':results,'generation':generation})
        (out/'report.partial.json').write_text(json.dumps(report,indent=2))
        del program,net,evaluator,future,donors
    report['status']='completed';report['decision']='Retain complete results; diagnostic evidence only, no future export or submission.'
    (out/'report.json').write_text(json.dumps(report,indent=2))
    append_event(events,'matched_horizon_audit_completed',evaluations=18)
    from finish_horizon_batch import finish
    finish('cnf_recent_detection_temporal_01','CNF_RECENT_DETECTION_TEMPORAL_RESULTS.json','cnf_recent_detection_temporal_job','800-update continuous-flow temporal checks complete. Retain all metrics and unchanged controls; no promotion without the original Monte Carlo and temporal gate. Review actual paired gains before declaring the next mechanism.')


if __name__=='__main__':
    try:
        torch.set_num_threads(2)
        with threadpool_limits(limits=2):main()
    except Exception as exc:
        failure=HERE/'private/cnf_recent_detection_temporal_01/failure.json'
        if failure.parent.exists():
            failure.write_text(json.dumps({'status':'execution_failed','error_type':type(exc).__name__,'error':str(exc),'resume_command':'outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/cnf_recent_detection_temporal.py --resume'},indent=2))
        raise
