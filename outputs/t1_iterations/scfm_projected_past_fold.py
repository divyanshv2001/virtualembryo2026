"""Fresh cutoff, matched scFM gradient-projection experiment; no official upload."""
import copy
import argparse
import json
import traceback
import subprocess
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from fresh_fullpanel_context import prepare, incremental_context
from frozen_fullpanel_scoring import score_frozen_forecasts
from anchored_soft_ot import train_anchored
from cnf_manifold_flow import DensityFlowNet, train_manifold_density
from full_anchor_slope_forecast import FullAnchorSlopeForecast
from temporary_forecast_cache import TemporaryForecastCache
from offline_backtest import load_core
from scnode_resource_preflight import peak_memory
from run_t1 import digest
from iterate import now, append_event

HERE=Path(__file__).resolve().parent
RUN=None
PUBLIC=None


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    global RUN,PUBLIC
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',default='scfm_projected_past_fold');args=parser.parse_args()
    entry=json.loads((HERE/'RESEARCH_HARNESS_MANIFEST.json').read_text())['experiments'][args.experiment]
    RUN=(HERE/entry['run']).resolve();PUBLIC=(HERE/entry['report']).resolve()
    if not RUN.is_relative_to(HERE/'private') or not PUBLIC.is_relative_to(HERE):raise ValueError('Evidence outside project')
    if RUN.exists() or PUBLIC.exists():raise ValueError('Never duplicate experiment')
    RUN.mkdir(parents=True);emit=lambda kind,**kw:append_event(RUN/'events.jsonl',kind,**kw)
    report={'status':'running','panels':[],'new_scoring_batch':True,'original_readiness_gate_passed':False,'official_score':None}
    cache=TemporaryForecastCache(HERE/'private/temporary_cache')
    try:
        spec=entry['protocol']
        prepared=HERE/'private/associated_prepared_01';prep=json.loads((prepared/'report.json').read_text())
        inputs={}
        for filename,key in [('expression.npy','expression_sha256'),('genes.csv','genes_sha256'),('selected_metadata.csv','metadata_sha256')]:
            inputs[filename]=digest(prepared/filename)
            if inputs[filename]!=prep[key]:raise ValueError('Prepared input changed')
        core,scorer=load_core()
        sources=['scfm_projected_past_fold.py','fresh_fullpanel_context.py','frozen_fullpanel_scoring.py','anchored_soft_ot.py','soft_ot_flow_matching.py','cnf_manifold_flow.py','cnf_density_flow.py','full_anchor_slope_forecast.py','log1p_positive_forecast.py','tigon_conditional_fullpanel.py','offline_backtest.py']
        if spec.get('learned_diffusion'): sources.append('learned_latent_diffusion.py')
        if spec.get('frozen_kinetic_run'): sources.append('graph_kinetic_residual.py')
        if spec.get('difference_attention'): sources.append('past_difference_attention.py')
        if spec.get('collective_context'): sources+=['collective_context_preflight.py','past_difference_attention.py']
        if spec.get('hidden_protein'): sources+=['hidden_protein_count_bridge.py','prepare_balanced_atlas.py']
        if spec.get('temporal_forcing'): sources.append('temporal_diffusion_forcing.py')
        if spec.get('count_vae'): sources.append('count_vae_representation.py')
        if spec.get('correlated_diffusion'): sources.append('correlated_latent_diffusion.py')
        if spec.get('tied_marginal_diffusion'): sources+=['tied_marginal_diffusion.py','correlated_latent_diffusion.py']
        if spec.get('scalar_potential'): sources.append('scalar_potential_residual.py')
        graph_prior_inputs={}
        if spec.get('graph_kinetics'):
            sources.append('graph_kinetic_residual.py')
            for name in ['adult_mouse_base_grn.parquet','past4096_adult_graph.npz','past4096_panel_features.npy','SOURCE_LICENSE.txt']:
                graph_prior_inputs[name]=digest(HERE/'private/flecs_adult_graph_01'/name)
        plan={'created_utc':now(),'protocol':spec,'source_sha256':{name:digest(HERE/name) for name in sources},'input_sha256':inputs,'scoring_seeds':spec['scoring_seeds'],'scorer_manifest':scorer,'submissions_allowed':0}
        if graph_prior_inputs:plan['graph_prior_sha256']=graph_prior_inputs
        save(RUN/'plan.json',plan);emit('plan_frozen',sha256=digest(RUN/'plan.json'))
        c=prepare(spec,RUN,emit);cutoff,target=spec['cutoff'],spec['target']
        if any(digest(HERE/name)!=expected for name,expected in plan['source_sha256'].items()):raise ValueError('Source changed after plan freeze')
        if len(np.flatnonzero(c['stages']==target))<2000:raise ValueError('Insufficient target metadata support')
        torch.manual_seed(spec['seed'])
        initial=DensityFlowNet(c['basis'],c['pca'].mean_,cutoff,float(c['stages'][c['past']].min())-.25)
        reuse=spec.get('reuse_run')
        if reuse:
            previous=HERE/reuse
            with np.load(previous/'fresh_encoder.npz') as old,np.load(RUN/'fresh_encoder.npz') as fresh:
                for key in old.files:np.testing.assert_array_equal(old[key],fresh[key])
            previous_plan=json.loads((previous/'plan.json').read_text())
            if previous_plan['input_sha256']!=inputs:raise ValueError('Reused input mismatch')
            saved=torch.load(previous/'cnf800.pt',weights_only=False,map_location='cpu')
            initial.load_state_dict(saved['net']);history=saved['history']
            if saved['steps']!=800 or saved['batch_size']!=64 or saved['density_weight']!=10.:raise ValueError('Reference metadata mismatch')
            initial.eval();baseline=copy.deepcopy(initial)
            saved=torch.load(previous/'projected_ot400.pt',weights_only=False,map_location='cpu')
            if saved['fit_max_stage']!=cutoff or saved['steps']!=400 or saved['batch_size']!=64:raise ValueError('Past projected checkpoint mismatch')
            baseline.load_state_dict(saved['net']);baseline.eval()
            plan['reused_checkpoint_sha256']={name:digest(previous/(name+'.pt')) for name in ['cnf800','projected_ot400']}
            save(RUN/'plan.json',plan);emit('reused_encoder_and_checkpoints_verified',sha256=digest(RUN/'plan.json'))
            flows={'cnf800':initial,'projected_ot400':baseline}
            if spec.get('frozen_kinetic_run'):
                from graph_kinetic_residual import KineticFlow
                frozen=HERE/spec['frozen_kinetic_run']
                frozen_plan=json.loads((frozen/'plan.json').read_text())
                if frozen_plan['input_sha256']!=inputs or frozen_plan['protocol']['cutoff']!=cutoff:raise ValueError('Frozen kinetic inputs changed')
                if digest(HERE/'graph_kinetic_residual.py')!=frozen_plan['source_sha256']['graph_kinetic_residual.py']:raise ValueError('Frozen kinetic mechanism changed')
                flows={'cnf800':initial}
                for name,expected in spec['frozen_checkpoint_sha256'].items():
                    path=frozen/(name+'.pt')
                    if digest(path)!=expected:raise ValueError('Frozen checkpoint hash mismatch')
                    saved=torch.load(path,weights_only=False,map_location='cpu')
                    if saved['steps']!=400 or saved['batch_size']!=64 or saved['fit_max_stage']!=cutoff or not saved['frozen_drift_exact']:raise ValueError('Frozen kinetic metadata mismatch')
                    state=saved['net'];edges=state['edges'].numpy()
                    model=KineticFlow(initial,state['gene_mean'].numpy(),edges[1],edges[0],saved['kind'])
                    model.load_state_dict(state)
                    for key,value in initial.state_dict().items():torch.testing.assert_close(model.drift.state_dict()[key],value,rtol=0,atol=0)
                    flows[name]=model.eval()
                report['fixed_candidate_validation']={'training_steps':0,'reward_eligible':False,'past_decoder_reconstruction':'Same deterministic past-only procedure; prior full-panel predictions must replay bit-exact before new horizon.'}
                emit('frozen_kinetic_checkpoints_verified',checkpoint_sha256=spec['frozen_checkpoint_sha256'])
                if spec.get('scalar_potential'):
                    from scalar_potential_residual import PotentialResidualFlow
                    report.pop('fixed_candidate_validation')
                    context_path=RUN/'residual_past_context.npz';np.savez_compressed(context_path,coordinates=c['coordinates'],stages=c['stages'][c['past']])
                    helper=HERE/'scalar_potential_residual.py';base_checkpoint=frozen/'kinetic_none400.pt'
                    packet={'context':str(context_path),'base_checkpoint':str(base_checkpoint),'sha256':{str(path):digest(path) for path in [context_path,base_checkpoint,helper,HERE/'cnf_manifold_flow.py',HERE/'cnf_density_flow.py',HERE/'graph_kinetic_residual.py']}}
                    save(RUN/'residual_GPU_packet.json',packet);python=HERE.parents[1]/'outputs/research_workflow/.venv_cuda/Scripts/python.exe'
                    with (RUN/'GPU_stdout.log').open('wb') as stdout,(RUN/'GPU_stderr.log').open('wb') as stderr:
                        subprocess.run([str(python),'-u',str(helper),'--train-packet',str(RUN/'residual_GPU_packet.json')],stdout=stdout,stderr=stderr,cwd=HERE.parents[1],check=True,creationflags=subprocess.CREATE_NO_WINDOW)
                    for kind in ['vector','potential']:
                        saved=torch.load(RUN/('residual_'+kind+'400.pt'),weights_only=False,map_location='cpu')
                        if saved['kind']!=kind or saved['steps']!=400 or saved['batch_size']!=64 or saved['fit_max_stage']!=cutoff or saved['parameters']!=1681 or saved['training_seed']!=20261004 or saved['step']!=.125 or saved['residual_scale']!=.1 or not saved['frozen_base_exact'] or saved['context_sha256']!=digest(context_path) or saved['base_checkpoint_sha256']!=digest(base_checkpoint):raise ValueError('Scalar residual checkpoint metadata mismatch')
                        model=PotentialResidualFlow(flows['kinetic_none400'],kind);model.residual_net.load_state_dict(saved['residual_net']);flows['residual_'+kind+'400']=model.eval()
                    report['scalar_potential_scope']=spec['scope']
                elif spec.get('tied_marginal_diffusion'):
                    from correlated_latent_diffusion import CorrelatedDiffusionFlow
                    from tied_marginal_diffusion import TiedMarginalDiffusionFlow
                    report.pop('fixed_candidate_validation')
                    prior=HERE/spec['amplitude_run'];context_path=prior/'diffusion_past_context.npz';amplitude_checkpoint=prior/'diffusion_diagonal400.pt';base_checkpoint=frozen/'kinetic_none400.pt';helper=HERE/'tied_marginal_diffusion.py'
                    old_plan=json.loads((prior/'plan.json').read_text())
                    if old_plan['input_sha256']!=inputs or old_plan['protocol']['cutoff']!=cutoff:raise ValueError('Amplitude inputs/cutoff changed')
                    for source in ['correlated_latent_diffusion.py','cnf_manifold_flow.py','cnf_density_flow.py','graph_kinetic_residual.py']:
                        if digest(HERE/source)!=old_plan['source_sha256'][source]:raise ValueError('Amplitude mechanism/solver changed')
                    if digest(context_path)!=spec['amplitude_context_sha256'] or digest(amplitude_checkpoint)!=spec['amplitude_checkpoint_sha256']:raise ValueError('Frozen amplitude/context changed')
                    with np.load(context_path) as old:
                        np.testing.assert_array_equal(old['coordinates'],c['coordinates']);np.testing.assert_array_equal(old['stages'],c['stages'][c['past']])
                    packet={'context':str(context_path),'base_checkpoint':str(base_checkpoint),'amplitude_checkpoint':str(amplitude_checkpoint),'sha256':{str(path):digest(path) for path in [context_path,base_checkpoint,amplitude_checkpoint,helper,HERE/'correlated_latent_diffusion.py',HERE/'cnf_manifold_flow.py',HERE/'cnf_density_flow.py',HERE/'graph_kinetic_residual.py']}}
                    save(RUN/'tied_GPU_packet.json',packet);python=HERE.parents[1]/'outputs/research_workflow/.venv_cuda/Scripts/python.exe'
                    with (RUN/'GPU_stdout.log').open('wb') as stdout,(RUN/'GPU_stderr.log').open('wb') as stderr:
                        subprocess.run([str(python),'-u',str(helper),'--train-packet',str(RUN/'tied_GPU_packet.json')],stdout=stdout,stderr=stderr,cwd=HERE.parents[1],check=True,creationflags=subprocess.CREATE_NO_WINDOW)
                    amplitude=torch.load(amplitude_checkpoint,weights_only=False,map_location='cpu');reference_noise=CorrelatedDiffusionFlow(flows['kinetic_none400'],'diagonal');reference_noise.noise_net.load_state_dict(amplitude['noise_net']);flows['diffusion_diagonal400']=reference_noise.eval()
                    for kind in ['diagonal','correlated']:
                        saved=torch.load(RUN/('tied_'+kind+'400.pt'),weights_only=False,map_location='cpu')
                        if saved['kind']!=kind or saved['steps']!=400 or saved['batch_size']!=64 or saved['fit_max_stage']!=cutoff or saved['parameters']!=816 or saved['training_seed']!=20261004 or saved['split_step']!=.125 or saved['normals_per_step']!=26 or not saved['frozen_base_amplitude_exact'] or saved['marginal_variance_max_error']>1e-8 or saved['context_sha256']!=digest(context_path) or saved['base_checkpoint_sha256']!=digest(base_checkpoint) or saved['amplitude_checkpoint_sha256']!=digest(amplitude_checkpoint):raise ValueError('Tied diffusion metadata mismatch')
                        model=TiedMarginalDiffusionFlow(flows['kinetic_none400'],reference_noise.noise_net,kind);model.factor_net.load_state_dict(saved['factor_net']);flows['tied_'+kind+'400']=model.eval()
                    report['tied_marginal_scope']=spec['scope']
                elif spec.get('correlated_diffusion'):
                    from correlated_latent_diffusion import CorrelatedDiffusionFlow
                    report.pop('fixed_candidate_validation')
                    weights_root=HERE/spec['reuse_diffusion_run'] if spec.get('reuse_diffusion_run') else RUN
                    context_path=weights_root/'diffusion_past_context.npz'
                    if spec.get('reuse_diffusion_run'):
                        old_plan=json.loads((weights_root/'plan.json').read_text())
                        if old_plan['input_sha256']!=inputs or old_plan['protocol']['cutoff']!=cutoff:raise ValueError('Frozen diffusion inputs/cutoff changed')
                        for source in ['correlated_latent_diffusion.py','cnf_manifold_flow.py','cnf_density_flow.py','graph_kinetic_residual.py']:
                            if digest(HERE/source)!=old_plan['source_sha256'][source]:raise ValueError('Frozen diffusion mechanism/solver changed')
                        if digest(context_path)!=spec['diffusion_context_sha256']:raise ValueError('Frozen diffusion context changed')
                        with np.load(context_path) as old:
                            np.testing.assert_array_equal(old['coordinates'],c['coordinates'])
                            np.testing.assert_array_equal(old['stages'],c['stages'][c['past']])
                        report['fixed_candidate_validation']={'training_steps':0,'reward_eligible':False,'scope':'Frozen diffusion weights, historical new horizon; no retraining, repeat reward or independent-embryo claim.'}
                    else:
                        np.savez_compressed(context_path,coordinates=c['coordinates'],stages=c['stages'][c['past']])
                    helper=HERE/'correlated_latent_diffusion.py';base_checkpoint=frozen/'kinetic_none400.pt'
                    packet={'context':str(context_path),'base_checkpoint':str(base_checkpoint),'sha256':{str(path):digest(path) for path in [context_path,base_checkpoint,helper,HERE/'cnf_manifold_flow.py',HERE/'cnf_density_flow.py',HERE/'graph_kinetic_residual.py']}}
                    plan['diffusion_context_sha256']=digest(context_path);save(RUN/'plan.json',plan)
                    if not spec.get('reuse_diffusion_run'):
                        save(RUN/'diffusion_GPU_packet.json',packet)
                        python=HERE.parents[1]/'outputs/research_workflow/.venv_cuda/Scripts/python.exe'
                        with (RUN/'GPU_stdout.log').open('wb') as stdout,(RUN/'GPU_stderr.log').open('wb') as stderr:
                            subprocess.run([str(python),'-u',str(helper),'--train-packet',str(RUN/'diffusion_GPU_packet.json')],stdout=stdout,stderr=stderr,cwd=HERE.parents[1],check=True,creationflags=subprocess.CREATE_NO_WINDOW)
                    for kind in ['diagonal','correlated']:
                        weight_path=weights_root/('diffusion_'+kind+'400.pt')
                        if spec.get('reuse_diffusion_run') and digest(weight_path)!=spec['diffusion_checkpoint_sha256']['diffusion_'+kind+'400']:raise ValueError('Frozen diffusion checkpoint changed')
                        saved=torch.load(weight_path,weights_only=False,map_location='cpu')
                        if saved['kind']!=kind or saved['steps']!=400 or saved['batch_size']!=64 or saved['fit_max_stage']!=cutoff or saved['parameters']!=1080 or saved['training_seed']!=20261004 or saved['split_step']!=.125 or saved['normals_per_step']!=26 or not saved['frozen_base_exact'] or saved['context_sha256']!=digest(context_path) or saved['base_checkpoint_sha256']!=digest(base_checkpoint):raise ValueError('Diffusion checkpoint metadata mismatch')
                        model=CorrelatedDiffusionFlow(flows['kinetic_none400'],kind);model.noise_net.load_state_dict(saved['noise_net'])
                        flows['diffusion_'+kind+'400']=model.eval()
                    report['correlated_diffusion_scope']='Own1080parameter residual SDE on frozen kinetics; diagonal vs rank2 correlated, matched initial marginal variance/paired26normal draws/400GPUsteps and past-only adjacent-stage Sinkhorn. FixedRK4/noise splitstep.125, seed20261004; independently trained arms need not retain equal marginal variances. No author solver reproduction or posthoc covariance grid. OriginalCPU fullpanel scorer/head/calibration/readiness gates,3scoringresamples,one trainingseed; historical development, not independent validation.'
                elif spec.get('count_vae'):
                    from count_vae_representation import ObservationVAE,LatentDynamics,VAELatentBridge
                    report.pop('fixed_candidate_validation')
                    context_path=HERE/'private/hidden_protein_count_bridge_repair_01/past_counts_context.npz'
                    if digest(context_path)!=spec['count_context_sha256']:raise ValueError('Count VAE context changed')
                    with np.load(context_path) as cc:
                        np.testing.assert_array_equal(c['features'][:128],cc['panel_features'])
                        np.testing.assert_allclose(cc['projection'],(c['basis'][:,:128]/c['scale'][:128]).T,rtol=0,atol=0)
                        projection=cc['projection'].copy()
                    helper=HERE/'count_vae_representation.py'
                    weights_root=RUN
                    if spec.get('reuse_vae_run'):
                        weights_root=HERE/spec['reuse_vae_run']
                        old_plan=json.loads((weights_root/'plan.json').read_text())
                        if old_plan['input_sha256']!=inputs or old_plan['protocol']['cutoff']!=cutoff:raise ValueError('Reused VAE input/fit cutoff mismatch')
                        for source in ['count_vae_representation.py','cnf_manifold_flow.py']:
                            if digest(HERE/source)!=old_plan['source_sha256'][source]:raise ValueError('Reused VAE model/solver changed')
                        report['fixed_candidate_validation']={'training_steps':0,'reward_eligible':False,'scope':'Frozen VAE new-horizon historical development; no retraining, target tuning or repeat reward.'}
                    else:
                        packet={'context':str(context_path),'training_seed':spec.get('vae_training_seed',20261004),'vae_kinds':spec.get('vae_kinds',['log_gaussian','count_nb']),'sha256':{str(path):digest(path) for path in [context_path,helper,HERE/'cnf_manifold_flow.py']}}
                        save(RUN/'VAE_GPU_packet.json',packet)
                        python=HERE.parents[1]/'outputs/research_workflow/.venv_cuda/Scripts/python.exe'
                        with (RUN/'GPU_stdout.log').open('wb') as stdout,(RUN/'GPU_stderr.log').open('wb') as stderr:
                            subprocess.run([str(python),'-u',str(helper),'--train-packet',str(RUN/'VAE_GPU_packet.json')],stdout=stdout,stderr=stderr,cwd=HERE.parents[1],check=True,creationflags=subprocess.CREATE_NO_WINDOW)
                    expected_z=flows['kinetic_none400'].encode(torch.tensor((c['donors'][:,c['features']]-c['center'])/c['scale']))[0].detach()
                    for kind in spec.get('vae_kinds',['log_gaussian','count_nb']):
                        name='vae_'+kind+'400';weight_path=weights_root/(name+'.pt')
                        if spec.get('reuse_vae_run') and digest(weight_path)!=spec['vae_checkpoint_sha256'][name]:raise ValueError('Frozen VAE checkpoint changed')
                        saved=torch.load(weight_path,weights_only=False,map_location='cpu')
                        if saved['kind']!=kind or saved['steps']!=400 or saved['dynamics_steps']!=400 or saved['vae_parameters']!=18320 or saved['dynamics_parameters']!=1640 or saved['fit_max_stage']!=cutoff or saved['batch_size']!=64 or saved['context_sha256']!=spec['count_context_sha256']:raise ValueError('VAE checkpoint metadata mismatch')
                        if saved.get('training_seed',20261004)!=spec.get('vae_training_seed',20261004):raise ValueError('VAE training seed mismatch')
                        model=ObservationVAE(kind);model.load_state_dict(saved['vae']);model.eval()
                        dynamics=LatentDynamics();dynamics.load_state_dict(saved['dynamics']);dynamics.eval()
                        flows[name]=VAELatentBridge(flows['kinetic_none400'],model,dynamics,saved['center'],saved['scale'],c['donors'][:,c['features'][:128]],projection,expected_z).eval()
                    report['count_vae_scope']='Own18320param NBcounts vslogGaussian observation VAE rank8/128genes, sameKL.01/backgroundcategory/past-only batches400GPUsteps; matched1640param latentOT400steps, whitening onpast posterior means. Decoder change relative to own cutoff reconstruction projected through frozenPCA normcap1 into retainedkinetics; originalCPU full-panel decoder/scorer/guards retained. Not scVI reproduction or isolated PCA comparison; likelihood units/dispersion semantics differ, one trainingseed/3scoringresamples.'
                    if spec.get('seed_confirmation'):
                        report['count_vae_scope']='Single fresh GaussianVAE trainingseed20261005 confirmation; unchanged18320VAE/1640latentOT parameters400+400GPUsteps and bridge. Frozen kinetics baseline, original CPU full-panel scorer/replays. Earlierseed20261004 evidence separate; no grid/independentembryo validation or repeat reward.'
                        report['seed_confirmation']={'training_seed':spec['vae_training_seed'],'previous_seed':20261004,'reward_eligible':False}
                elif spec.get('temporal_forcing'):
                    from temporal_diffusion_forcing import TemporalDenoiser,TemporalBridgeFlow
                    report.pop('fixed_candidate_validation')
                    context_path=RUN/'temporal_past_context.npz'
                    np.savez_compressed(context_path,coordinates=c['coordinates'],stages=c['stages'][c['past']])
                    helper=HERE/'temporal_diffusion_forcing.py'
                    packet={'context':str(context_path),'sha256':{str(path):digest(path) for path in [context_path,helper]}}
                    save(RUN/'temporal_GPU_packet.json',packet)
                    python=HERE.parents[1]/'outputs/research_workflow/.venv_cuda/Scripts/python.exe'
                    with (RUN/'GPU_stdout.log').open('wb') as stdout,(RUN/'GPU_stderr.log').open('wb') as stderr:
                        subprocess.run([str(python),'-u',str(helper),'--train-packet',str(RUN/'temporal_GPU_packet.json')],stdout=stdout,stderr=stderr,cwd=HERE.parents[1],check=True,creationflags=subprocess.CREATE_NO_WINDOW)
                    expected_z=flows['kinetic_none400'].encode(torch.tensor((c['donors'][:,c['features']]-c['center'])/c['scale']))[0].detach()
                    earlier=c['coordinates'][c['stages'][c['past']]==8.]
                    for kind in ['conditional','forcing']:
                        name='temporal_'+kind+'400';saved=torch.load(RUN/(name+'.pt'),weights_only=False,map_location='cpu')
                        if saved['kind']!=kind or saved['steps']!=400 or saved['parameters']!=9224 or saved['fit_max_stage']!=cutoff or saved['batch_size']!=64 or saved['context_sha256']!=digest(context_path):raise ValueError('Temporal checkpoint metadata mismatch')
                        model=TemporalDenoiser();model.load_state_dict(saved['net']);model.eval()
                        flows[name]=TemporalBridgeFlow(flows['kinetic_none400'],model,earlier,expected_z).eval()
                    report['temporal_forcing_scope']='Own causal3token PCA8 denoising: clean-history conditional vs independently noised history, same final-token x0loss9224parameters400GPUsteps. Ordered independently sampled pastcells, not individual lineages. Fixed100diffusionlevels/20DDIMsteps/quarter-day autoregressive rollout with past8.0context and actual8.25donors, normcap1 correction to frozenkinetic trajectory. Original CPU full-panel decoder/guards/scorer retained; not CellPace/scVI reproduction, no gap embedding/fullsequence loss. Stage extrapolation and generated-history shift unvalidated.'
                elif spec.get('hidden_protein'):
                    from hidden_protein_count_bridge import CountProteinFlow,CountBridgeFlow
                    report.pop('fixed_candidate_validation')
                    context_path=HERE/'private/hidden_protein_count_bridge_repair_01/past_counts_context.npz'
                    if digest(context_path)!=spec['count_context_sha256']:raise ValueError('Frozen past count context changed')
                    helper=HERE/'hidden_protein_count_bridge.py'
                    packet={'context':str(context_path),'sha256':{str(path):digest(path) for path in [context_path,helper,HERE/'prepare_balanced_atlas.py']}}
                    save(RUN/'count_GPU_packet.json',packet)
                    plan['count_context_sha256']=digest(context_path);save(RUN/'plan.json',plan)
                    python=HERE.parents[1]/'outputs/research_workflow/.venv_cuda/Scripts/python.exe'
                    with (RUN/'GPU_stdout.log').open('wb') as stdout,(RUN/'GPU_stderr.log').open('wb') as stderr:
                        subprocess.run([str(python),'-u',str(helper),'--train-packet',str(RUN/'count_GPU_packet.json')],stdout=stdout,stderr=stderr,cwd=HERE.parents[1],check=True,creationflags=subprocess.CREATE_NO_WINDOW)
                    with np.load(context_path) as cc:
                        np.testing.assert_array_equal(c['features'][:128],cc['panel_features'])
                        np.testing.assert_allclose(cc['projection'],(c['basis'][:,:128]/c['scale'][:128]).T,rtol=0,atol=0)
                        import pandas as pd
                        metadata=pd.read_csv(prepared/'selected_metadata.csv')
                        source_rows=metadata.source_row.to_numpy()[np.load(RUN/'donor_rows.npy')]
                        lookup={int(value):i for i,value in enumerate(cc['source_rows'])}
                        selected=np.array([lookup[int(value)] for value in source_rows])
                        reference_count=float(cc['library_reference'])
                        anchor_counts=cc['counts'][selected]*(reference_count/cc['libraries'][selected,None])
                        projection=cc['projection'].copy()
                    expected_z=flows['kinetic_none400'].encode(torch.tensor((c['donors'][:,c['features']]-c['center'])/c['scale']))[0].detach()
                    for kind in ['instantaneous','delayed']:
                        name='protein_'+kind+'400';saved=torch.load(RUN/(name+'.pt'),weights_only=False,map_location='cpu')
                        if saved['kind']!=kind or saved['steps']!=400 or saved['parameters']!=4480 or saved['fit_max_stage']!=cutoff or saved['batch_size']!=64 or saved['context_sha256']!=spec['count_context_sha256']:raise ValueError('Count checkpoint metadata mismatch')
                        model=CountProteinFlow(128,kind);model.load_state_dict(saved['net']);model.eval()
                        flows[name]=CountBridgeFlow(flows['kinetic_none400'],model,anchor_counts,projection,reference_count,expected_z).eval()
                    report['hidden_protein_scope']='Own128gene lowrank NB-mixture RNA production: instantaneous vs fixed1/day delayed protein; matched4480params/400GPUsteps. Original count library exposure offsets; first128 frozen pastfeature genes, no target selection. Countchanges projected through frozenencoder with normcap1 and added to frozenkinetic trajectory; original CPU full-panel head/guards/scorer retained. Not author CardamomOT reproduction; independent-gene NB assumptions disclosed.'
                elif spec.get('collective_context'):
                    from collective_context_preflight import CollectiveContextFlow
                    report.pop('fixed_candidate_validation')
                    context_path=RUN/'collective_training_context.npz'
                    np.savez_compressed(context_path,coordinates=c['coordinates'],stages=c['stages'][c['past']],basis=c['basis'],pca_center=c['pca'].mean_)
                    helper=HERE/'collective_context_preflight.py';base_checkpoint=frozen/'kinetic_none400.pt'
                    packet={'context':str(context_path),'kinetic_checkpoint':str(base_checkpoint),'sha256':{str(path):digest(path) for path in [context_path,base_checkpoint,helper,HERE/'past_difference_attention.py',HERE/'cnf_density_flow.py',HERE/'graph_kinetic_residual.py']}}
                    save(RUN/'collective_GPU_packet.json',packet)
                    python=HERE.parents[1]/'outputs/research_workflow/.venv_cuda/Scripts/python.exe'
                    with (RUN/'GPU_stdout.log').open('wb') as stdout,(RUN/'GPU_stderr.log').open('wb') as stderr:
                        subprocess.run([str(python),'-u',str(helper),'--train-packet',str(RUN/'collective_GPU_packet.json')],stdout=stdout,stderr=stderr,cwd=HERE.parents[1],check=True,creationflags=subprocess.CREATE_NO_WINDOW)
                    for name,kind in [('collective_frozen400','frozen'),('collective_evolving400','evolving')]:
                        saved=torch.load(RUN/(name+'.pt'),weights_only=False,map_location='cpu')
                        if saved['kind']!=kind or saved['steps']!=400 or saved['batch_size']!=64 or saved['parameters']!=4776 or not saved['frozen_base_exact']:raise ValueError('Collective checkpoint metadata mismatch')
                        model=CollectiveContextFlow(flows['kinetic_none400'],kind);model.load_state_dict(saved['net']);model.eval()
                        for key,value in flows['kinetic_none400'].state_dict().items():torch.testing.assert_close(model.base.state_dict()[key],value,rtol=0,atol=0)
                        flows[name]=model
                    report['collective_scope']='Own frozeninitial versus evolving population context, equal4776parameters/400RTX3060steps. Original2.14CPU forecast/head/scorer; fixed256context cap/128query chunk/self-exclusion. No causal communication or author reproduction claim.'
                if spec.get('difference_attention'):
                    from past_difference_attention import PastAttentionFlow,train_attention
                    times=np.unique(c['stages'][c['past']])
                    centroids=np.stack([c['coordinates'][c['stages'][c['past']]==t].mean(0) for t in times])
                    report.pop('fixed_candidate_validation')
                    report['attention_scope']='Own past-centroid level versus interval-normalized difference attention. Same4776parameters/frozenkinetics/CUDA400step budget; no individual-cell histories or author reproduction.'
                    for name,kind in [('attention_level400','level'),('attention_difference400','difference')]:
                        torch.manual_seed(20261004)
                        model=PastAttentionFlow(flows['kinetic_none400'],times,centroids,kind)
                        if spec.get('reuse_attention_run'):
                            prior_run=HERE/spec['reuse_attention_run'];path=prior_run/(name+'.pt')
                            if digest(path)!=spec['attention_checkpoint_sha256'][name]:raise ValueError('Attention checkpoint changed')
                            old_plan=json.loads((prior_run/'plan.json').read_text())
                            if old_plan['input_sha256']!=inputs or old_plan['source_sha256']['past_difference_attention.py']!=digest(HERE/'past_difference_attention.py'):raise ValueError('Attention training provenance changed')
                            saved=torch.load(path,weights_only=False,map_location='cpu')
                            if saved['kind']!=kind or saved['steps']!=400 or saved['batch_size']!=64 or saved['fit_max_stage']!=cutoff or not saved['frozen_base_exact'] or saved['cuda']!='12.8':raise ValueError('Attention training metadata changed')
                            model.load_state_dict(saved['net']);model.eval()
                            for key,value in flows['kinetic_none400'].state_dict().items():torch.testing.assert_close(model.base.state_dict()[key],value,rtol=0,atol=0)
                            np.testing.assert_array_equal(model.centroids.numpy(),centroids)
                            emit('CUDA_trained_attention_checkpoint_reused',candidate=name,sha256=digest(path),optimizer_steps=0)
                        else:
                            train_attention(model,c['coordinates'],c['stages'][c['past']],RUN/(name+'.pt'),emit,steps=spec['steps'],batch=spec['batch_size'])
                        flows[name]=model;emit('candidate_training_finished',candidate=name,device='cuda:0',reused=bool(spec.get('reuse_attention_run')))
            elif spec.get('graph_kinetics'):
                from graph_kinetic_residual import prepare_kinetic_models, train_kinetics
                kinetic_models,graph_audit=prepare_kinetic_models(initial,c,RUN,emit)
                report['kinetic_graph_audit']=graph_audit
                flows={'cnf800':initial}
                for name,model in kinetic_models.items():
                    train_kinetics(model,c['coordinates'],c['stages'][c['past']],RUN/(name+'.pt'),emit,steps=spec['steps'],batch=spec['batch_size'])
                    flows[name]=model.eval();emit('candidate_training_finished',candidate=name)
                report['kinetic_scope']='FLeCS-inspired approximate reconstructed-gene kinetic residual in frozen PCA/CNF/decoder. Real vs degree-preserving shuffled graph is primary; no-edge has fewer parameters. Possible adult binding, not signed causal regulation or author reproduction. Past physical stages only.'
            elif spec.get('learned_diffusion'):
                from learned_latent_diffusion import DiffusionFlow, train_diffusion, smoke_test
                report['diffusion_preflight']=smoke_test(initial)
                emit('diffusion_preflight_passed', **report['diffusion_preflight'])
                flows={'cnf800':initial}
                for name,kind in [('constant_diffusion400','constant'),('state_diffusion400','state')]:
                    torch.manual_seed(spec['diffusion_seed'])
                    model=DiffusionFlow(initial,kind,seed=spec['diffusion_seed'])
                    train_diffusion(model,c['coordinates'],c['stages'][c['past']],RUN/(name+'.pt'),emit,steps=spec['steps'],batch=spec['batch_size'])
                    flows[name]=model.eval();emit('candidate_training_finished',candidate=name)
                report['diffusion_scope']='Frozen past CNF drift/heads; learned diagonal diffusion; scDiffEq-inspired CPU adaptation, not author reproduction. Constant vs state dependence isolates mechanism; original CNF uses different RK4 step. No future loss, no growth inference.'
            elif spec.get('fixed_candidate_validation'):
                report['fixed_candidate_validation']={'training_steps':0,'reward_eligible':False,'reason':'New horizon evaluation of already-accounted frozen candidate; no repeat model reward.'}
            elif spec.get('representation_comparison'):
                ci=incremental_context(c,RUN,emit,chunk=spec['ipca_chunk'])
                torch.manual_seed(spec['seed'])
                ipca=DensityFlowNet(ci['basis'],ci['pca'].mean_,cutoff,float(ci['stages'][ci['past']].min())-.25)
                train_manifold_density(ipca,ci['coordinates'],ci['stages'][ci['past']],.1,RUN/'ipca_cnf800.pt',emit,steps=800,density_weight=10.)
                flows['ipca_cnf800']=ipca.eval();model=copy.deepcopy(ipca)
                train_anchored(model,ci['coordinates'],ci['stages'][ci['past']],cutoff,RUN/'ipca_projected400.pt',emit,conflict_projection=True)
                flows['ipca_projected400']=model.eval()
                report['representation_diagnostics']={'same_fit_rows':True,'fit_rows':3000,'features':4096,'rank':8,'ipca_chunk':spec['ipca_chunk'],'past_fit_explained_variance_ratio_sum':{'pca':float(c['pca'].explained_variance_ratio_.sum()),'ipca':float(ci['pca'].explained_variance_ratio_.sum())},'interpretation':'Fit diagnostic only, not benchmark score or independent validation.'}
            else:
                for name,weight in [('projected_ot800',0.),('projected_pair800',1.)]:
                    model=copy.deepcopy(baseline)
                    train_anchored(model,c['coordinates'],c['stages'][c['past']],cutoff,RUN/(name+'.pt'),emit,conflict_projection=True,latent_variogram_weight=weight)
                    flows[name]=model.eval();emit('candidate_training_finished',candidate=name)
        else:
            history=train_manifold_density(initial,c['coordinates'],c['stages'][c['past']],.1,RUN/'cnf800.pt',emit,steps=800,density_weight=10.)
            initial.eval();flows={'cnf800':initial}
            for name in ['likelihood400','raw_ot400','projected_ot400']:
                model=copy.deepcopy(initial)
                if name=='likelihood400':
                    train_manifold_density(model,c['coordinates'],c['stages'][c['past']],.1,RUN/(name+'.pt'),emit,steps=400,density_weight=10.)
                else:
                    train_anchored(model,c['coordinates'],c['stages'][c['past']],cutoff,RUN/(name+'.pt'),emit,conflict_projection=name=='projected_ot400')
                flows[name]=model.eval();emit('candidate_training_finished',candidate=name)
        reference=FullAnchorSlopeForecast(c['x'],c['stages'],cutoff,c['donors'],c['panel'],c['symbols'],initial,c['center'],c['scale'],c['features'],c['guard'],anchor_path=c['anchor_path'])
        reference.configure(.25,.75)
        reference.audit.update(fit_max_stage=cutoff,method='Fresh source-domain PCA8 paired scFM',training_history=history)
        references={name:reference for name in flows}
        if spec.get('representation_comparison'):
            ipca_reference=FullAnchorSlopeForecast(ci['x'],ci['stages'],cutoff,ci['donors'],ci['panel'],ci['symbols'],flows['ipca_cnf800'],ci['center'],ci['scale'],ci['features'],ci['guard'],anchor_path=ci['anchor_path'])
            ipca_reference.configure(.25,.75)
            ipca_reference.audit.update(fit_max_stage=cutoff,method='Fresh source-domain IncrementalPCA8 paired scFM')
            for name in ['ipca_cnf800','ipca_projected400']:references[name]=ipca_reference
        generation={}
        if spec.get('frozen_kinetic_run'):
            previous_generation=json.loads((HERE/spec['frozen_kinetic_run']/'generation.json').read_text())
            for name in spec.get('prior_candidates',spec['candidates']):
                if name=='copy':prior=c['donors'].copy()
                else:
                    reference.net=flows[name]
                    prior,_,_=reference.predict(spec['prior_target'],'joint',1.,sampling='systematic')
                if cache.put('_prior_'+name,prior)!=previous_generation[name]['prediction_sha256']:raise ValueError('Original kinetic forecast reconstruction changed')
                del prior
            emit('prior_horizon_forecasts_reconstructed_bit_exact',target=spec['prior_target'])
        if spec.get('reuse_diffusion_run'):
            previous_generation=json.loads((HERE/spec['reuse_diffusion_run']/'generation.json').read_text())
            for name in spec['candidates']:
                if name=='copy':prior=c['donors'].copy()
                else:
                    reference.net=flows[name];prior,_,_=reference.predict(spec['diffusion_prior_target'],'joint',1.,sampling='systematic')
                if cache.put('_diffusion_prior_'+name,prior)!=previous_generation[name]['prediction_sha256']:raise ValueError('Frozen diffusion prior full-panel replay mismatch')
                del prior
            emit('frozen_diffusion_prior_fullpanel_replay_passed',target=spec['diffusion_prior_target'],candidates=spec['candidates'])
        if spec.get('hidden_protein') or spec.get('temporal_forcing') or spec.get('count_vae'):
            reference.net=flows['kinetic_none400'];neutral,_,_=reference.predict(target,'joint',1.,sampling='systematic')
            neutral_sha=cache.put('_neutral_kinetic',neutral);del neutral
            for name in spec['contrast_candidates']:
                reference.net=flows[name];reference.net.bridge_enabled=False
                neutral,_,_=reference.predict(target,'joint',1.,sampling='systematic')
                if cache.put('_neutral_'+name,neutral)!=neutral_sha:raise ValueError('Zero-count correction full-panel replay mismatch')
                del neutral;reference.net.bridge_enabled=True
                emit('neutral_count_bridge_fullpanel_exact_replay_passed',candidate=name)
        if spec.get('tied_marginal_diffusion'):
            old_generation=json.loads((HERE/spec['amplitude_run']/'generation.json').read_text())
            reference.net=flows['diffusion_diagonal400'];prior,_,_=reference.predict(9.5,'joint',1.,sampling='systematic')
            if cache.put('_prior_amplitude_diagonal',prior)!=old_generation['diffusion_diagonal400']['prediction_sha256']:raise ValueError('Frozen amplitude forecast changed')
            del prior;emit('frozen_amplitude_fullpanel_prior_exact_replay_passed')
        if spec.get('scalar_potential'):
            reference.net=flows['kinetic_none400'];neutral,_,_=reference.predict(target,'joint',1.,sampling='systematic')
            neutral_sha=cache.put('_neutral_kinetic',neutral);del neutral
            for name in spec['contrast_candidates']:
                reference.net=flows[name];reference.net.residual_enabled=False
                neutral,_,_=reference.predict(target,'joint',1.,sampling='systematic')
                if cache.put('_neutral_'+name,neutral)!=neutral_sha:raise ValueError('Zero-residual full-panel replay mismatch')
                del neutral;reference.net.residual_enabled=True;emit('neutral_potential_residual_fullpanel_exact_replay_passed',candidate=name)
        if spec.get('correlated_diffusion') or spec.get('tied_marginal_diffusion'):
            reference.net=flows['kinetic_none400'];neutral,_,_=reference.predict(target,'joint',1.,sampling='systematic')
            neutral_sha=cache.put('_neutral_kinetic',neutral);del neutral
            for name in spec['contrast_candidates']:
                reference.net=flows[name];reference.net.noise_enabled=False
                neutral,_,_=reference.predict(target,'joint',1.,sampling='systematic')
                if cache.put('_neutral_'+name,neutral)!=neutral_sha:raise ValueError('Zero-noise full-panel replay mismatch')
                del neutral;reference.net.noise_enabled=True
                emit('neutral_diffusion_fullpanel_exact_replay_passed',candidate=name)
        for name in spec['candidates']:
            if name=='copy':pred,ids,audit=c['donors'].copy(),np.arange(len(c['donors'])),{'method':'persistence'}
            else:
                reference=references[name];reference.net=flows[name];pred,ids,audit=reference.predict(target,'joint',1.,sampling='systematic')
            np.testing.assert_array_equal(pred[:,c['protected']],c['donors'][ids][:,c['protected']])
            if not np.isfinite(pred).all() or (pred<0).any():raise ValueError('Invalid forecast')
            sha=cache.put(name,pred);del pred
            if name=='copy':replay,replay_ids=c['donors'].copy(),np.arange(len(c['donors']))
            else:replay,replay_ids,_=reference.predict(target,'joint',1.,sampling='systematic')
            if cache.put('_replay_'+name,replay)!=sha:raise ValueError('Exact replay failed')
            np.testing.assert_array_equal(ids,replay_ids);del replay
            np.save(RUN/(name+'_indices.npy'),ids)
            generation[name]={'prediction_sha256':sha,'audit':copy.deepcopy(audit)}
            emit('forecast_frozen_exact_replay_passed',candidate=name,prediction_sha256=sha)
        save(RUN/'generation.json',generation);report['generation']=generation
        if any(digest(HERE/name)!=expected for name,expected in plan['source_sha256'].items()):raise ValueError('Source changed before target scoring')
        emit('all_forecasts_frozen_before_target_expression',target=target)
        score_frozen_forecasts(core,c['raw_x'],c['stages'],c['donors'],c['mapped'],c['columns'],c['panel'],cutoff,target,plan,spec['candidates'],generation,cache,RUN,report,emit,spec)
        report[spec.get('contrast_label','projected_minus_raw_ot_mean_skills')]=report.pop('growth_enabled_minus_disabled_mean_skills')
    except Exception as exc:
        (RUN/'traceback.txt').write_text(traceback.format_exc());report.update(status='failed',error=type(exc).__name__+': '+str(exc),failed_utc=now(),passing_candidates=[])
    finally:cache.close()
    report['resource_peak_process_working_set_bytes']=peak_memory()
    if peak_memory()>=16*1024**3:report.update(status='failed',error='16GiB resource gate failed',passing_candidates=[])
    save(RUN/'report.json',report);report['report_sha256']=digest(RUN/'report.json');save(PUBLIC,report)
    emit('batch_finished',status=report['status'],report_sha256=report['report_sha256'])
    print(json.dumps({'status':report['status'],'report':str(PUBLIC),'error':report.get('error')}))


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
