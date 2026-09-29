"""User-authorized progress export, refitted at observed E9.5 for E10.5."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import anndata as ad
from scipy import sparse
from threadpoolctl import threadpool_limits
from train_extended_atlas import HERE
from run_t1 import digest,validate
from iterate import now,append_event
from hurdle_backtest import read_cells
from past_encoder_panel import fit_past_encoder
from cnf_manifold_flow import DensityFlowNet,train_manifold_density
from anchor_slope_calibration import AnchorSlopeCalibration


def main():
    root=HERE.parents[1];name='anchorslope_progress_20260929_01'
    out=root/'outputs/t1_submissions'/name;private=HERE/'private'/name
    if out.exists() or private.exists():raise ValueError('Preserve prior export; use a new version')
    data=HERE/'private/associated_prepared_01'
    prepared=json.loads((data/'report.json').read_text())
    for f,k in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(data/f)!=prepared[k]:raise ValueError('Prepared data changed')
    panel_path=root/'outputs/t1_run/T1__val.genes.txt';panel=panel_path.read_text().splitlines()
    anchor=root/'data/E9.5_RNA.h5ad';index_path=root/'outputs/t1_run/index.json'
    development=HERE/'CNF_ANCHOR_SLOPE_RESULTS.json'
    candidate=next(s for s in json.loads(development.read_text())['summaries'] if s['candidate']=='anchorslope_both_0.5')
    sources=['export_anchor_slope_submission.py','past_encoder_panel.py','anchor_slope_calibration.py','feature_panel_forecast.py','ridge_conditional_head.py','cnf_manifold_flow.py','cnf_density_flow.py','hurdle_backtest.py','neural_hurdle_forecast.py','robust_population.py']
    plan={'created_before_training_utc':now(),'authorization':'User requested a submission file to check the score. Export only; no upload.',
        'observed_cutoff':9.5,'forecast_target':10.5,'source_max_stage':9.5,'source_cohort':'associated_prepared_01',
        'candidate':'anchorslope_both_0.5','seed':20260928,'donor_count':1500,'encoder_genes':4096,'dimensions':8,
        'training_steps':800,'batch_size':64,'energy_weight':.1,'density_weight':10.,'positive_alpha':.5,'detection_alpha':.5,
        'input_sha256':{'anchor':digest(anchor),'panel':digest(panel_path),'index':digest(index_path),'prepared_report':digest(data/'report.json')},
        'source_sha256':{f:digest(HERE/f) for f in sources},'development_report_sha256':digest(development),
        'local_development':candidate,'readiness_gate_passed':False,'official_score':None,'uploaded':False,
        'scope':'Refit past-only encoder, flow and conditional heads on sampled external atlas throughE9.5; adapt slopes using1500 observed challengeE9.5 anchors. PredictE10.5 without hidden expression. Local54.91 does not estimate official score. Temporal checks pending at request.',
        'external_data_disclosure':'Development2024 dev201867 associated prepared cohort:25963 cells selected from430339-cell public atlas, CC-BY. Not full atlas training.',
        'guard_features_sha256':digest(HERE/'private/transport_challenge_01/features.npy')}
    out.mkdir(parents=True);private.mkdir(parents=True);(out/'plan.json').write_text(json.dumps(plan,indent=2))
    for f in sources:(private/f).write_bytes((HERE/f).read_bytes())
    events=out/'trajectory.jsonl';append_event(events,'progress_export_plan_frozen',plan_sha256=digest(out/'plan.json'))
    x=np.load(data/'expression.npy',mmap_mode='r');stages=pd.read_csv(data/'selected_metadata.csv').numeric_stage.to_numpy(float)
    if not np.isfinite(stages).all() or stages.max()>9.5:raise ValueError('Source cutoff violation')
    symbols=pd.read_csv(data/'genes.csv').symbol.fillna('').tolist()
    donors,rows=read_cells(anchor,panel,1500,20260928);np.save(private/'observed_rows.npy',rows)
    append_event(events,'observed_inputs_loaded',source_shape=list(x.shape),anchor_shape=list(donors.shape),maximum_stage=float(stages.max()))
    encoded=fit_past_encoder(x,stages,9.5,symbols,panel,4096,8)
    np.savez_compressed(private/'encoder.npz',**encoded)
    append_event(events,'past_only_encoder_fitted',encoder_sha256=digest(private/'encoder.npz'),past_cells=len(encoded['past_rows']))
    torch.manual_seed(20260928);net=DensityFlowNet(encoded['basis'],encoded['pca_center'],9.5,float(stages[encoded['past_rows']].min())-.25)
    history=train_manifold_density(net,encoded['coordinates'],stages[encoded['past_rows']],.1,private/'training.pt',lambda kind,**kw:append_event(events,kind,**kw),steps=800,density_weight=10.)
    guards=np.load(HERE/'private/transport_challenge_01/features.npy')
    model=AnchorSlopeCalibration(x,stages,9.5,donors,panel,symbols,net,encoded['center'],encoded['scale'],encoded['features'],guards)
    model.configure(.5,.5)
    np.savez_compressed(private/'heads.npz',positive_coef=model.positive_coef,detection=model.detection,positive_delta=model.anchor_positive_delta,detection_delta=model.anchor_detection_delta,positive_center=model.anchor_positive_center,latent_mean=model.anchor_latent_mean,support=model.support)
    append_event(events,'past_only_model_and_anchor_slopes_fitted',checkpoint_sha256=digest(private/'training.pt'),heads_sha256=digest(private/'heads.npz'))
    pred,indices,audit=model.predict(10.5,'joint',1.,sampling='systematic');np.save(private/'donor_indices.npy',indices)
    protected=np.setdiff1d(np.arange(len(panel)),model.mapped)
    exact=bool(np.array_equal(pred[:,protected],donors[indices][:,protected]))
    before=np.expm1(donors[indices][:,model.mapped].astype(float)).sum(1);after=np.expm1(pred[:,model.mapped].astype(float)).sum(1)
    error=float(np.max(abs(after-before)/np.maximum(before,1e-12)))
    if not exact or error>1e-5:raise ValueError('Protected gene or mass guard failed')
    append_event(events,'E10_5_forecast_generated',shape=list(pred.shape),audit=audit,protected_genes_exact=exact,mapped_mass_max_relative_error=error)
    artifact=out/'T1_val__anchorslope_progress_20260929_01.h5ad'
    a=ad.AnnData(sparse.csr_matrix(pred.astype(np.float32)),obs=pd.DataFrame(index=[f'forecast_{i:05d}' for i in range(len(pred))]),var=pd.DataFrame(index=panel))
    a.uns['forecast_target_stage']='E10.5';a.uns['observed_cutoff_stage']='E9.5';a.uns['purpose']='User-requested progress export; local readiness unmet'
    a.write_h5ad(artifact,compression='gzip')
    validation=validate(artifact,panel,json.loads(index_path.read_text())['T1:val'])
    if not validation['passed']:raise ValueError('Submission format failed')
    report={'artifact':artifact.name,'plan_sha256':digest(out/'plan.json'),'format_validation':validation,'forecast_audit':audit,'training_history':history,'protected_genes_exact':exact,'mapped_mass_max_relative_error':error,'local_development_mean':candidate['mean_score'],'readiness_gate_passed':False,'official_score':None,'uploaded':False,'external_data_disclosure':plan['external_data_disclosure'],'evidence_scope':'Genuine trajectory records this export only, not the entire research history.'}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    (out/'README.md').write_text(f'# T1 E10.5 progress submission\n\nUpload `{artifact.name}` for T1:val. 1500 cells, exact32285gene order, finite nonnegativefloat32, no coordinates. Refitted using observedE9.5 and sampled public source atlas throughE9.5.\n\nLocal E8.5-to-E9.5 development mean54.91 is not an official E10.5 score. Local>72 gate unmet; no upload performed. Disclose the external Development2024 dev201867 atlas. Export trajectory is genuine but covers this export only.\n')
    append_event(events,'progress_submission_validated',artifact_sha256=digest(artifact),validation=validation)
    print(json.dumps({'artifact':str(artifact),'validation':validation,'local_score':candidate['mean_score']}))


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
