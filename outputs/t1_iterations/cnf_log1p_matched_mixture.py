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
from cell_mixture import mix_matched_cells
from sklearn.decomposition import PCA
from ridge_conditional_head import RidgeConditionalPositiveForecast
from transport_anchor_alignment import predict_aligned
from offline_backtest import load_core,Panel


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    root=HERE.parents[1];data=HERE/'private/associated_prepared_01';out=HERE/'private/cnf_log1p_matched_mixture_01'
    if out.exists() and not args.resume:raise ValueError('Preserve prior run; use --resume')
    prepared=json.loads((data/'report.json').read_text())
    for f,k in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(data/f)!=prepared[k]:raise ValueError('Prepared data changed')
    anchor_path=root/'data/E8.5_RNA.h5ad';target_path=root/'data/E9.5_RNA.h5ad'
    panel=(root/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
    controls=['copy','anchorslope_both_0.5','log1p_abundance_a0.5','log1p_joint_a0.5']
    configs=controls+[f'matched_log1p_{weight}' for weight in [.25,.5,.75]]
    old=HERE/'private/cnf_feature_challenge_01'
    encoder_root=HERE/'private/transport_challenge_01'
    plan={'created_before_training_utc':now(),'configs':configs,'evaluation_seeds':[20260928,20260929,20260930],
        'cutoff':8.5,'target':9.5,'donor_seed':20260928,'donor_count':1500,'truth_count':1000,'ceiling_count':1000,
        'source_sha256':{f:digest(HERE/f) for f in ['cnf_log1p_matched_mixture.py','cell_mixture.py','transport_anchor_alignment.py','hurdle_backtest.py','hurdle_transfer.py','detection_transfer.py','challenge_transfer.py','robust_population.py','constrained_forecast.py','transfer_genes.py','feature_panel_forecast.py','windowed_positive_head.py','cnf_manifold_flow.py','cnf_density_flow.py','dimensional_transport_flow.py','transport_latent_flow.py','ridge_conditional_head.py','neural_hurdle_forecast.py','offline_backtest.py']},
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
    plan['hypothesis']='Independent source-row mixtures can duplicate/omit donor identities, altering population composition. Matched mixtures keep every donor once, choose one complete constituent forecast per donor, and isolate this sampling mechanism with fixedweights/seed and prior mixtures ascontrols.'
    plan['fit']='No new biological training. Reuse exact frozen past-trained log1p_abundance_a0.5 and log1p_joint_a0.5 outputs, verified donor identity and source hashes. Mixture selection uses exposed development, and both constituents fail first temporal fold; this experiment cannot promote them.'
    plan['scope']='Ten constituent/incumbent/persistence/previous-mixture controls plus three matched complete-cell mixtures, weights.25/.5/.75 and one fixed rowselection seed2026092812, three unchanged E9.5 development panels=39scores. Source and all rows retained; no fresh validation or independent embryos.'
    plan['mixture_weights']=[.25,.5,.75];plan['mixture_seed']=2026092812
    plan['strength']=1.;plan['latent_velocity_cap']=None;plan['training_steps']=0
    plan['manifold_fit']='Constituent past-only fits remain frozen; no field or decoder retraining in mixture experiment.'
    plan['encoder_sha256']=digest(old/'encoder4096.npz');plan['heads_sha256']=digest(old/'features4096_heads.npz')
    plan['fit']='No refit. Frozen past-only log1p positive abundance and joint forecasts, eachsame1500 donors. Fixedoneforecastperdonor mixtures withseed2026092812.'
    plan['hypothesis']='Abundance-only improves localDE53.317 butjoint has betterdirection57.765 andheadline54.9755. Test componenttradeoffintegration without combiningpergene metricoptima or changing evaluator.'
    plan['scope']='Three matchedwholecell mixtures abundanceweights.25/.5/.75 plus fourcontrols xthreeunchangedpanels=21scores. No independent validation/embryos; no promotion withoutoriginalgate.'
    def control_path(name):
        run='cnf_anchor_slope_challenge_01' if name=='anchorslope_both_0.5' else 'cnf_log1p_positive_challenge_01'
        return HERE/'private'/run/(name+'.npy')
    plan['constituent_report_sha256']={n:digest(control_path(n).parent/'report.json') for n in ['log1p_abundance_a0.5','log1p_joint_a0.5']}
    plan['archived_report_sha256']=digest(old/'report.json')
    plan['archived_features_sha256']=digest(old/'features.npy')
    plan['archived_control_sha256']={n:digest(control_path(n)) for n in controls}
    if args.resume:
        previous=json.loads((out/'plan.json').read_text())
        for k in ['configs','mixture_weights','mixture_seed','constituent_report_sha256','evaluation_seeds','source_sha256','input_sha256','prepared_report_sha256','panel_sha256','dependencies','archived_report_sha256','archived_features_sha256','archived_control_sha256','author_reference_sha256','past_encoder_sha256','manifold_reference_sha256','reference_checkpoint_sha256','encoder_sha256','heads_sha256']:
            if previous[k]!=plan[k]:raise ValueError('Resume mismatch:'+k)
        plan=previous
    else:
        out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2))
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
        first_path=control_path('log1p_abundance_a0.5');second_path=control_path('log1p_joint_a0.5')
        for source in [first_path,second_path]:
            np.testing.assert_array_equal(donor_rows,np.load(source.parent/'anchor_rows.npy'))
        first=np.load(first_path);second=np.load(second_path)
        if first.shape!=donors.shape or second.shape!=donors.shape:raise ValueError('Constituent full-panel mismatch')
        first_indices=np.load(first_path.with_name('log1p_abundance_a0.5_indices.npy'));second_indices=np.load(second_path.with_name('log1p_joint_a0.5_indices.npy'))
        np.testing.assert_array_equal(first_indices,second_indices)
        np.testing.assert_array_equal(np.sort(first_indices),np.arange(len(donors)))
        for weight in plan['mixture_weights']:
            prediction,origin=mix_matched_cells(first,second,weight,plan['mixture_seed'])
            indices=np.where(origin[:,0]==0,first_indices[origin[:,1]],second_indices[origin[:,1]])
            name=f'matched_log1p_{weight}'
            np.save(out/(name+'_provenance.npy'),origin)
            audit={'method':'Matched mixture of complete frozen specialist cells, one forecast per donor','abundance_weight':weight,'seed':plan['mixture_seed'],
                   'whole_cell_provenance_sha256':digest(out/(name+'_provenance.npy')),'abundance_cell_count':int((origin[:,0]==0).sum()),
                   'joint_cell_count':int((origin[:,0]==1).sum()),'protected_genes_and_mapped_mass':'Inherited unchanged from each complete constituent cell and its donor index; no gene-wise averaging.',
                   'limitations':'Exposed development mixture selection; bothconstituents lack temporalreadiness. No promotion or official export.'}
            if prediction.dtype!=np.float32 or not np.isfinite(prediction).all() or (prediction<0).any():raise ValueError('Invalid mixture')
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
    (HERE/'CNF_LOG1P_MATCHED_MIXTURE_RESULTS.json').write_text(json.dumps(public,indent=2))
    p=HERE/'LOCAL_OPTIMIZATION_STATE.json';state=json.loads(p.read_text());state['cnf_log1p_matched_mixture_job'].update(status='completed_not_promoted',evaluations=21,pending_evaluations=0,report_sha256=public['report_sha256'],results_report='CNF_LOG1P_MATCHED_MIXTURE_RESULTS.json');state['local_process_running']=False;state['active_jobs']=[];p.write_text(json.dumps(state,indent=2))
    p=HERE/'METRIC_RESEARCH_QUEUE.json';queue=json.loads(p.read_text())
    for entry in queue['paths']:
        if entry.get('run')=='cnf_log1p_matched_mixture_01':entry.update(status='implemented_evaluated_not_promoted',results_report='CNF_LOG1P_MATCHED_MIXTURE_RESULTS.json')
    p.write_text(json.dumps(queue,indent=2))
    from index_scores import main as index_scores
    index_scores();append_event(events,'challenge_transport_completed',evaluations=21)
    from update_compact_checkpoint import refresh
    refresh('Completed21 matched log1p specialist mixture scores; retain fullmetrics in CNF_LOG1P_MATCHED_MIXTURE_RESULTS.json. Review constituent tradeoffs, no official upload or72 certification.')


if __name__=='__main__':
    try:
        torch.set_num_threads(2)
        with threadpool_limits(limits=2):main()
    except Exception as exc:
        folder=HERE/'private/cnf_log1p_matched_mixture_01'
        if folder.exists():(folder/'failure.json').write_text(json.dumps({'status':'execution_failed','type':type(exc).__name__,'error':str(exc),'resume_command':'outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/cnf_log1p_matched_mixture.py --resume'},indent=2))
        raise
