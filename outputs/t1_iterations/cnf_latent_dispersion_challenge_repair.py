"""Past-only transport transfer diagnostic on exposed challenge development data."""
import argparse
import json
import importlib.metadata
from collections import Counter
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now,append_event
from hurdle_backtest import read_cells
from challenge_transfer import ChallengeTransfer
from dimensional_transport_flow import DimensionalTransportFlow
from cnf_manifold_flow import DensityFlowNet,train_manifold_density
from feature_panel_forecast import FeaturePanelForecast
from log1p_positive_forecast import Log1pPositiveForecast
from latent_dispersion import LatentDispersion
from sklearn.decomposition import PCA
from ridge_conditional_head import RidgeConditionalPositiveForecast
from transport_anchor_alignment import predict_aligned
from offline_backtest import load_core,Panel


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    root=HERE.parents[1];data=HERE/'private/associated_prepared_01';out=HERE/'private/cnf_latent_dispersion_challenge_repair_01'
    if out.exists() and any(out.iterdir()) and not args.resume:raise ValueError('Preserve prior run; use --resume')
    prepared=json.loads((data/'report.json').read_text())
    for f,k in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(data/f)!=prepared[k]:raise ValueError('Prepared data changed')
    anchor_path=root/'data/E8.5_RNA.h5ad';target_path=root/'data/E9.5_RNA.h5ad'
    panel=(root/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
    controls=['copy','anchorslope_both_0.5','log1p_abundance_a0.5','log1p_joint_a0.5']
    configs=controls+[f'dispersion_{mode}_{sigma}' for mode in ['abundance','joint'] for sigma in [.025,.05,.1]]
    old=HERE/'private/cnf_feature_challenge_01'
    encoder_root=HERE/'private/transport_challenge_01'
    plan={'created_before_training_utc':now(),'configs':configs,'evaluation_seeds':[20260928,20260929,20260930],
        'cutoff':8.5,'target':9.5,'donor_seed':20260928,'donor_count':1500,'truth_count':1000,'ceiling_count':1000,
        'source_sha256':{f:digest(HERE/f) for f in ['cnf_latent_dispersion_challenge_repair.py','anchor_slope_calibration.py','log1p_positive_forecast.py','latent_dispersion.py','transport_anchor_alignment.py','hurdle_backtest.py','hurdle_transfer.py','detection_transfer.py','challenge_transfer.py','robust_population.py','constrained_forecast.py','transfer_genes.py','feature_panel_forecast.py','windowed_positive_head.py','cnf_manifold_flow.py','cnf_density_flow.py','dimensional_transport_flow.py','transport_latent_flow.py','ridge_conditional_head.py','neural_hurdle_forecast.py','offline_backtest.py']},
        'input_sha256':{p.name:digest(p) for p in [anchor_path,target_path]},'prepared_report_sha256':digest(data/'report.json'),
        'panel_sha256':digest(root/'outputs/t1_run/T1__val.genes.txt'),
        'dependencies':{k:importlib.metadata.version(k) for k in ['numpy','scipy','torch','scikit-learn','anndata']},
        'fit':'Same13963 permitted prepared atlas rows<=E8.5 and archived384 features. Archived8D past PCA, whitened using all permitted past coordinate variances. Gaussian base at7.25 before first prepared7.5 snapshot. Time-conditioned3x64 tanh field; fixedRK4.125 and one Rademacher trace vector per minibatch64,400 Adam.001 updates, gradient cap5. All-past full conditional positive ridge1 heads; abundance cap2/covariance.4/mapped mass/protected genes unchanged. No latent velocity cap or biological growth/velocity input.',
        'scope':'Eight candidates across three unchanged frozen challengeE9.5 development panels;24 evaluations. Observation noise0/.05/.1 on whitened latent training cells; separate fixed RNG preserves stage/trace sampling. Density10/energy.1,800 updates fixed. No independent embryo validation; E9.5 previously exposed. Missing/ambiguous atlas genes preserve actual challenge donor values. Complete32285-gene scorer and original calibration unchanged.',
        'hypothesis':'Past-manifold density regularization may discourage unsupported transition paths. Compare midpoint5-neighbor Euclidean hinge at.1 with weights1/10 versus0, energy.1 fixed,400 updates and strengths.5/1. References are all permitted past cells, never future. Own adaptation, not full TrajectoryNet reproduction.',
        'gate':'No official submission or E10.5 export. Three panels cannot satisfy>=64 Monte Carlo gate. Mean/lower-tail>72 and temporal consistency still required. Retain every raw metric, skill and invalid calibration.',
        'manifold_fit':'Five-neighbor Euclidean hinge.1 at half-quarter inverse midpoint for past stages after earliest. All past latent cells as reference; energy.1 fixed; density weights0/1/10. Zero density model must replay prior400-step training exactly.',
        'strength':.5,'latent_velocity_cap':6,'abundance_factor_cap':2,'covariance_limit':.4,'submissions_allowed':0,'jev_requests_allowed':0}
    plan['past_encoder_sha256']=digest(encoder_root/'neutral_d8_flow.npz')
    plan['author_reference_sha256']=digest(HERE/'CNF_AUTHOR_REFERENCE.json')
    plan['manifold_reference_sha256']=digest(HERE/'CNF_MANIFOLD_REFERENCE.json')
    plan['reference_checkpoint_sha256']=digest(old/'features4096.pt')
    plan['hypothesis']='Atlas conditional slopes may mismatch the anchor domain. Fit anchor ridge1 slopes, interpolate .5/1, preserve source levels at anchor centroids; separate and joint mechanisms, previous intercept controls.'
    plan['fit']='Original4096-gene8D800-update field/encoder/source heads fixed. Conditional positive and linear probability ridge1 slopes from1500 observed E8.5 anchors, support20 and original source support. Center corrections at conditional positive centroids/detection latent mean. No target learning or temporal velocity inference.'
    plan['scope']='Six slope variants plus ten controls including previous intercept variants x3 unchanged E9.5 panels=48 full-panel development scores. Cross-sectional associations need not transfer as temporal directions.'
    plan['mechanisms']=['abundance','joint'];plan['dispersion_scales']=[.025,.05,.1];plan['dispersion_seed']=2026092917
    plan['anchor_reference_sha256']=digest(HERE/'ANCHOR_SLOPE_CALIBRATION_REFERENCE.json')
    plan['strength']=1.;plan['latent_velocity_cap']=None;plan['training_steps']=0
    plan['manifold_fit']='Original800-updatefield frozen; only conditional slopes calibrated on observed past anchors; source levels centered unchanged.'
    plan['encoder_sha256']=digest(old/'encoder4096.npz');plan['heads_sha256']=digest(old/'features4096_heads.npz')
    plan['hypothesis']='Changing positive response from log(raw) to log1p(raw) may improve scored DE while retaining latentflow/detection and rawspace output guards.'
    plan['fit']='Fixedoriginal4096gene8D800step CNF, fitpositive conditional ridge1 in log1p units on<=E8.5 cells. Anchorpositive slope strength0/.5; anchor detection.5 fixed; abundance/joint decoding. Own metric-scale adaptation, no futurefit.'
    plan['scope']='Fourresponse variants plus threearchives x3 matched panels=21 scores. Full32285genes/unchanged calibration. Allconditionalpositive centroids retained, mass/protected/covariance guards unchanged. Reused development, no readiness pass.'
    plan['hypothesis']='Deterministic forecast spread may mismatch expression heterogeneity; fixedterminal dispersion sensitivity can test variance effects with no learned diffusion/extrapolation claim.'
    plan['fit']='Exactfrozen4096gene8Dflow andlog1p conditionalridge heads;positive/detectionanchoralpha.5. Zero dispersion must replay bothconstituents exactly. Add antitheticterminal Gaussian sigma*sqrt(duration) usingfixedseed2026092917, no futurefit.'
    plan['scope']='Six terminaldispersion abundance/joint variants .025/.05/.1 plusfourcontrols x3unchangedE9.5panels=30scores. Full32285genes andoriginal floor/ceiling. Not scDiffusion reproduction or measured diffusion rate; no independent validation.'
    plan['dispersion_reference_sha256']=digest(HERE/'LATENT_DISPERSION_REFERENCE.json')
    plan['storage_repair']={'prior_failed_run':'private/cnf_latent_dispersion_challenge_01','prior_plan_sha256':digest(HERE/'private/cnf_latent_dispersion_challenge_01/plan.json'),'reason':'DiskfullD, failedarray retained. NewrunE storage via localjunction; all10forecasts regenerated under same seeds/configs before targetread.'}
    def control_path(name):return HERE/'private/cnf_log1p_positive_challenge_01'/(name+'.npy')
    plan['archived_control_sha256']={n:digest(control_path(n)) for n in controls}
    if args.resume:
        previous=json.loads((out/'plan.json').read_text())
        for k in ['configs','mechanisms','dispersion_scales','dispersion_seed','dispersion_reference_sha256','anchor_reference_sha256','evaluation_seeds','source_sha256','input_sha256','prepared_report_sha256','panel_sha256','dependencies','archived_report_sha256','archived_features_sha256','archived_control_sha256','author_reference_sha256','past_encoder_sha256','manifold_reference_sha256','reference_checkpoint_sha256','encoder_sha256','heads_sha256']:
            if previous[k]!=plan[k]:raise ValueError('Resume mismatch:'+k)
        plan=previous
    else:
        out.mkdir(exist_ok=True);(out/'plan.json').write_text(json.dumps(plan,indent=2))
        for f in plan['source_sha256']:(out/f).write_bytes((HERE/f).read_bytes())
    events=out/'events.jsonl';append_event(events,'challenge_transport_plan_resumed' if args.resume else 'challenge_transport_plan_frozen',sha256=digest(out/'plan.json'))
    x=np.load(data/'expression.npy',mmap_mode='r');stages=pd.read_csv(data/'selected_metadata.csv').numeric_stage.to_numpy(float)
    symbols=pd.read_csv(data/'genes.csv').symbol.fillna('').tolist()
    donors,donor_rows=read_cells(anchor_path,panel,1500,plan['donor_seed']);np.save(out/'anchor_rows.npy',donor_rows)
    generation_path=out/'generation.json'
    if not generation_path.exists():
        features=np.load(old/'features.npy');np.save(out/'features.npy',features)
        np.testing.assert_array_equal(donor_rows,np.load(old/'anchor_rows.npy'))
        old_generation=json.loads((old/'generation.json').read_text());old_generation['joint_systematic_s1.0']=json.loads((HERE/'private/cnf_hurdle_challenge_01/generation.json').read_text())['joint_systematic_s1.0'];generation={}
        def save(name,pred,indices,audit):
            np.save(out/(name+'.npy'),pred);np.save(out/(name+'_indices.npy'),indices)
            generation[name]={'audit':audit,'prediction_sha256':digest(out/(name+'.npy'))}
            append_event(events,'challenge_transport_forecast_frozen',candidate=name,prediction_sha256=generation[name]['prediction_sha256'])
        for name in controls:
            source=control_path(name);pred=np.load(source);indices=np.load(source.with_name(name+'_indices.npy'))
            if digest(source)!=plan['archived_control_sha256'][name]:raise ValueError('Control changed')
            save(name,pred,indices,{**json.loads((source.parent/'generation.json').read_text())[name]['audit'],'replayed_from':str(source)})
        guard_features=features.copy()
        encoder=np.load(old/'encoder4096.npz');encoder_features=encoder['features']
        center=encoder['center'];scale=encoder['scale']
        net=DensityFlowNet(encoder['basis'],encoder['pca_center'],8.5,7.25)
        saved=torch.load(old/'features4096.pt',weights_only=False,map_location='cpu');net.load_state_dict(saved['net']);net.eval()
        program=Log1pPositiveForecast(x,stages,8.5,donors,panel,symbols,net,center,scale,encoder_features,guard_features)
        append_event(events,'past_only_log1p_response_heads_fitted',fit_max_stage=8.5,model_audit=program.audit)
        np.savez_compressed(out/'hurdle_heads.npz',detection=program.detection,pmean=program.pmean,positive_coef=program.positive_coef,positive_center=program.positive_center,positive_mean=program.positive_mean,zcenter=program.zcenter,support=program.support)
        np.savez_compressed(out/'anchor_offsets.npz',positive_delta=program.anchor_positive_delta,detection_delta=program.anchor_detection_delta,positive_center=program.anchor_positive_center,latent_mean=program.anchor_latent_mean,count=program.anchor_count,support=program.calibration_support)
        program.configure(.5,.5)
        spread=LatentDispersion(net,0.,plan['dispersion_seed']);program.net=spread
        for mode in plan['mechanisms']:
            np.testing.assert_array_equal(program.predict(9.5,mode,1.,sampling='systematic')[0],np.load(control_path(f'log1p_{mode}_a0.5')))
        append_event(events,'zero_dispersion_replays_both_archived_forecasts_exactly')
        for mode in plan['mechanisms']:
            for sigma in plan['dispersion_scales']:
                spread.configure(sigma,plan['dispersion_seed'])
                name=f'dispersion_{mode}_{sigma}'
                prediction,indices,audit=program.predict(9.5,mode,1.,sampling='systematic')
                audit.update(terminal_latent_dispersion=sigma,dispersion_seed=plan['dispersion_seed'],dispersion_policy='Fixed antithetic Gaussian terminal perturbation sigma*sqrt(duration) in whitened latent units; not a learned diffusion model, calibrated biological rate or multi-seed stability result.')
                save(name,prediction,indices,audit)
        if set(generation)!=set(configs):raise ValueError('Forecast coverage mismatch')
        generation_path.write_text(json.dumps(generation,indent=2))
        append_event(events,'all_challenge_forecasts_frozen_before_target_read',candidates=configs)
    generation=json.loads(generation_path.read_text())
    for name in configs:
        if digest(out/(name+'.npy'))!=generation[name]['prediction_sha256']:raise ValueError('Frozen forecast changed')
    core,_=load_core();report={'plan':plan,'generation':generation,'panels':[],'official_score':None,'local_gate_passed':False,'scorer_manifest_sha256':digest(HERE/'private/scorer_source/manifest.json')}
    if args.resume and (out/'report.partial.json').exists():report=json.loads((out/'report.partial.json').read_text());report.pop('in_progress_panel',None)
    for seed in plan['evaluation_seeds']:
        if any(p['seed']==seed for p in report['panels']):continue
        future,rows=read_cells(target_path,panel,2000,seed);np.save(out/f'target_rows_{seed}.npy',rows)
        order=np.random.default_rng(seed).permutation(len(future));evaluator=Panel(core,future[order[:1000]],donors,seed)
        floor=evaluator.metrics(donors);ceiling=evaluator.metrics(future[order[1000:]])
        previous=next(p for p in json.loads((old/'report.json').read_text())['panels'] if p['seed']==seed)
        if floor!=previous['floor'] or ceiling!=previous['ceiling']:raise ValueError('Calibration panels changed')
        results=[]
        for name in configs:
            raw=evaluator.metrics(np.load(out/(name+'.npy'),mmap_mode='r'))
            result={'candidate':name,'prediction_sha256':generation[name]['prediction_sha256'],'raw_metrics':raw,**evaluator.aggregate(raw,floor,ceiling)}
            results.append(result);append_event(events,'challenge_transport_candidate_scored',seed=seed,**result)
            (out/'report.partial.json').write_text(json.dumps({**report,'in_progress_panel':{'seed':seed,'cutoff':8.5,'target':9.5,'floor':floor,'ceiling':ceiling,'results':results}},indent=2))
        report['panels'].append({'seed':seed,'cutoff':8.5,'target':9.5,'floor':floor,'ceiling':ceiling,'results':results})
        (out/'report.partial.json').write_text(json.dumps(report,indent=2))
    for path in [anchor_path,target_path]:
        if digest(path)!=plan['input_sha256'][path.name]:raise ValueError('Challenge input changed')
    report['status']='completed';(out/'report.json').write_text(json.dumps(report,indent=2))
    summaries=[]
    for name in configs:
        results=[next(r for r in p['results'] if r['candidate']==name) for p in report['panels']]
        valid=all(r['calibration_valid'] for r in results);scores=[r['local_score'] for r in results]
        summaries.append({'candidate':name,'scores':scores,'mean_score':float(np.mean(scores)) if valid else None,'all_calibrations_valid':valid,'raw_metrics':[r['raw_metrics'] for r in results],'skills':[r['skills'] for r in results]})
    public={'updated_utc':now(),'status':'completed_not_promoted','summaries':summaries,'panels':report['panels'],'plan_sha256':digest(out/'plan.json'),'report_sha256':digest(out/'report.json'),'scope':plan['scope'],'local_72_gate_passed':False,'official_score':None,'submissions_used':0}
    (HERE/'CNF_LATENT_DISPERSION_RESULTS.json').write_text(json.dumps(public,indent=2))
    p=HERE/'LOCAL_OPTIMIZATION_STATE.json';state=json.loads(p.read_text());state['cnf_latent_dispersion_repair_job'].update(status='completed_not_promoted',evaluations=30,pending_evaluations=0,report_sha256=public['report_sha256'],results_report='CNF_LATENT_DISPERSION_RESULTS.json');state['local_process_running']=False;state['active_jobs']=[];p.write_text(json.dumps(state,indent=2))
    p=HERE/'METRIC_RESEARCH_QUEUE.json';queue=json.loads(p.read_text())
    for entry in queue['paths']:
        if entry.get('run')=='cnf_latent_dispersion_challenge_repair_01':entry.update(status='implemented_evaluated_not_promoted',results_report='CNF_LATENT_DISPERSION_RESULTS.json')
    p.write_text(json.dumps(queue,indent=2))
    from index_scores import main as index_scores
    index_scores();append_event(events,'challenge_transport_completed',evaluations=30)
    from update_compact_checkpoint import refresh
    refresh('Completed30 terminal latent dispersion scores. All means/raw4/skills4 retained in CNF_LATENT_DISPERSION_RESULTS.json and SCORE_LEDGER.jsonl. Review matched gains before promotion;72 gate remains unmet.')


if __name__=='__main__':
    try:
        torch.set_num_threads(2)
        with threadpool_limits(limits=2):main()
    except Exception as exc:
        folder=HERE/'private/cnf_latent_dispersion_challenge_repair_01'
        if folder.exists():(folder/'failure.json').write_text(json.dumps({'status':'execution_failed','type':type(exc).__name__,'error':str(exc),'resume_command':'outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/cnf_latent_dispersion_challenge_repair.py --resume'},indent=2))
        raise
