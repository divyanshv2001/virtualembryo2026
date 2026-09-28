"""User-requested E10.5 progress export; never implies research promotion."""
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from detection_transfer import DetectionTransfer
from state_search import read_cells
from train_extended_atlas import HERE
from run_t1 import digest, validate
from iterate import append_event, now


def main():
    root = HERE.parents[1]
    name = 'unit16_progress_20260928_01'
    out = root/'outputs/t1_submissions'/name
    private = HERE/'private'/name
    if out.exists() or private.exists():
        raise ValueError('Preserve existing export; use a new version for another run')
    data = HERE/'private/associated_prepared_01'
    prepared = json.loads((data/'report.json').read_text())
    for filename, key in [('expression.npy','expression_sha256'),
                          ('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(data/filename) != prepared[key]:
            raise ValueError('Prepared source changed')
    panel_path = root/'outputs/t1_run/T1__val.genes.txt'
    panel = panel_path.read_text().splitlines()
    index_path = root/'outputs/t1_run/index.json'
    spec = json.loads(index_path.read_text())['T1:val']
    anchor = root/'data/E9.5_RNA.h5ad'
    validation = HERE/'private/state_search_01/report.json'
    stability = HERE/'private/monte_carlo_states_01/report.json'
    sources = ['export_progress_submission.py','state_search.py','detection_transfer.py',
               'challenge_transfer.py','robust_population.py','constrained_forecast.py','transfer_genes.py']
    plan = {'created_before_training_utc':now(), 'purpose':'User-requested progress submission',
        'authorization':'User asked to generate a submission file after the experimental batch.',
        'readiness_gate_passed':False, 'official_score':None, 'upload_authorized':False,
        'gate_override':'Export only; does not waive local >72, temporal or stability promotion gates.',
        'source_dataset':'associated_prepared_01', 'source_max_stage':9.5,
        'observed_challenge_stage':9.5, 'forecast_target_stage':10.5,
        'config':{'states':16,'alignment':'identity','feature_scaling':'unit',
                  'covariance_limit':.4,'expression_factor_cap':1.25,
                  'detection':.5,'expression':1.,'detection_cap':.02},
        'donor_count':1500,'donor_seed':20260928,
        'selection':'Retain previous incumbent: broadened source and importance variants underperformed on identical local panels.',
        'input_sha256':{'observed_E9.5':digest(anchor),'panel_file':digest(panel_path),
                        'index':digest(index_path),'prepared_report':digest(data/'report.json')},
        'validation_report_sha256':digest(validation),'stability_report_sha256':digest(stability),
        'source_sha256':{f:digest(HERE/f) for f in sources},
        'fit_scope':'Atlas <=E9.5 and observed challenge E9.5 only; no E10.5 truth or hidden scoring anchors.',
        'score_scope':'Earlier reused E8.5-to-E9.5 local development; not an official E10.5 estimate.'}
    out.mkdir(parents=True); private.mkdir(parents=True)
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    for f in sources: (private/f).write_bytes((HERE/f).read_bytes())
    events = out/'trajectory.jsonl'
    append_event(events,'progress_export_plan_frozen',plan_sha256=digest(out/'plan.json'))
    x = np.load(data/'expression.npy',mmap_mode='r')
    stages = pd.read_csv(data/'selected_metadata.csv').numeric_stage.to_numpy(float)
    if not np.isfinite(stages).all() or stages.max() > 9.5:
        raise ValueError('Source exceeds observed cutoff')
    symbols = pd.read_csv(data/'genes.csv').symbol.fillna('').tolist()
    donors, rows = read_cells(anchor,panel,1500,20260928)
    np.save(private/'observed_rows.npy',rows)
    append_event(events,'observed_inputs_loaded',atlas_shape=list(x.shape),donor_shape=list(donors.shape),
                 maximum_atlas_stage=float(stages.max()))
    with threadpool_limits(limits=2):
        model = DetectionTransfer(x,stages,9.5,donors,panel,symbols,states=16,
            alignment='identity',covariance_limit=.4,feature_scaling='unit',expression_factor_cap=1.25)
        model.save(private/'model.npz')
        append_event(events,'past_only_model_fitted',model_sha256=digest(private/'model.npz'))
        prediction, indices, audit = model.predict_detection(10.5,.5,1.,.02)
    protected = np.setdiff1d(np.arange(len(panel)),model.mapped)
    protected_exact = bool(np.array_equal(prediction[:,protected],donors[indices][:,protected]))
    mapped_mass_before = np.expm1(donors[indices][:,model.mapped].astype(float)).sum(1)
    mapped_mass_after = np.expm1(prediction[:,model.mapped].astype(float)).sum(1)
    mass_error = float(np.max(np.abs(mapped_mass_after-mapped_mass_before)/np.maximum(mapped_mass_before,1e-12)))
    if not protected_exact or mass_error > 1e-5:
        raise ValueError('Protected genes or mapped mass guard failed')
    np.save(private/'donor_indices.npy',indices)
    append_event(events,'E10_5_forecast_generated',shape=list(prediction.shape),audit=audit,
                 protected_genes_exact=protected_exact,mapped_mass_max_relative_error=mass_error)
    artifact = out/'T1_val__unit16_progress_20260928_01.h5ad'
    a = ad.AnnData(sparse.csr_matrix(prediction.astype(np.float32)),
        obs=pd.DataFrame(index=[f'forecast_{i:05d}' for i in range(len(prediction))]),
        var=pd.DataFrame(index=panel))
    a.uns['forecast_target_stage']='E10.5'
    a.uns['observed_cutoff_stage']='E9.5'
    a.uns['purpose']='User-requested progress candidate; local >72 readiness gate unmet'
    a.write_h5ad(artifact,compression='gzip')
    result = validate(artifact,panel,spec)
    if not result['passed']: raise ValueError('Submission format failed')
    report = {'plan_sha256':digest(out/'plan.json'),'artifact':artifact.name,
        'format_validation':result,'forecast_audit':audit,
        'protected_gene_count':len(protected),'protected_genes_exact':protected_exact,
        'mapped_mass_max_relative_error':mass_error,
        'local_development_mean':55.312268380984996,
        'local_stability_pilot':{'replicates':16,'mean':55.28130592055242,
                               'empirical_2_5_percentile':52.60162180056054},
        'readiness_gate_passed':False,'official_score':None,'uploaded':False,
        'evidence_scope':'Trajectory records this export run only, not the complete historical Agent Team research workflow.',
        'limitations':['Repeated local development and no independent embryo interval.',
                       'No same-configuration temporal promotion or >72 stability pass.',
                       'E10.5 extrapolation remains unverified by hidden ground truth.']}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    (out/'README.md').write_text(
        '# T1 E10.5 progress submission\n\n'
        f'Upload candidate: `{artifact.name}`. This is an actual E10.5 forecast refitted on observed E9.5, not an E9.5 backtest. '
        'It contains 1,500 cells and the exact 32,285-gene panel, with float32 finite nonnegative expression and no coordinate embeddings.\n\n'
        'The user requested this progress export. The >72 promotion gate remains unmet: prior local development mean 55.31; '
        '16-replicate stability mean 55.28 and empirical 2.5th percentile 52.60. These are local E9.5 results, not an official E10.5 score. '
        'New broad-cohort and importance variants scored about 51.6 and did not replace the incumbent.\n\n'
        'No upload was performed. Submitting consumes a daily submission slot. The report records format checks, artifact SHA256 and model guards; '
        'the frozen plan records inputs and code hashes. The trajectory is the genuine execution record for this export only. '
        'It is not a complete Agent Team research trajectory or a certification of evidence eligibility.\n')
    append_event(events,'progress_submission_validated',artifact_sha256=digest(artifact),validation=result)
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
