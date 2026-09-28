"""Diagnose confirmed official failure using only already observed E8.5/E9.5."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from state_search import read_cells
from train_extended_atlas import HERE
from offline_backtest import load_core
from run_t1 import digest
from iterate import now, append_event


def main():
    root=HERE.parents[1]
    folder=root/'outputs/t1_submissions/unit16_progress_20260928_01'
    result_path=folder/'official_result.json'
    artifact=folder/'T1_val__unit16_progress_20260928_01.h5ad'
    export=json.loads((folder/'report.json').read_text())
    if digest(artifact)!=export['format_validation']['sha256']: raise ValueError('Artifact changed')
    result={'recorded_utc':now(),'source':'User-provided official result screenshot and explicit filename confirmation',
        'source_image':'codex-clipboard-19627395-4782-448c-9907-0676205ac371.png',
        'artifact':artifact.name,'artifact_sha256':digest(artifact),'headline_score':46.59,
        'metric_skills_percent':{'de_score':38.2,'de_direction':51.3,'mmd_u':48.6,'variogram':48.2},
        'rank':217,'board_entries':268,'screenshot_metric_rounding':True,
        'score_verified_via_authenticated_connector':False,
        'decision':'Failed official task; below persistence baseline. Do not promote or resubmit unchanged.'}
    if result_path.exists(): raise ValueError('Preserve previously recorded result')
    result_path.write_text(json.dumps(result,indent=2))
    append_event(folder/'trajectory.jsonl','official_result_received_from_user',
        headline_score=46.59,result_sha256=digest(result_path),source='Screenshot and filename confirmation after export')
    panel=(root/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
    early,rows8=read_cells(root/'data/E8.5_RNA.h5ad',panel,1500,20260928)
    later,rows9=read_cells(root/'data/E9.5_RNA.h5ad',panel,1500,20260928)
    # Match the first declared development panel exactly; target is known development data.
    future,rows=read_cells(root/'data/E9.5_RNA.h5ad',panel,2000,20260928)
    future=future[np.random.default_rng(20260928).permutation(2000)[:1000]]
    core,manifest=load_core()
    up,down,_=core.de_genes(future,early)
    with np.load(HERE/'private/unit16_progress_20260928_01/model.npz') as model:
        mapped=model['mapped']; protected=np.setdiff1d(np.arange(len(panel)),mapped)
    truth_de=np.union1d(up,down)
    local=json.loads((HERE/'private/state_search_01/generation.json').read_text())['unit_k16']
    official_audit=export['forecast_audit']
    pred=np.load(HERE/'private/state_search_01/unit_k16.npy',mmap_mode='r')
    true_delta=future.mean(0)-early.mean(0)
    pred_delta=np.asarray(pred).mean(0)-early.mean(0)
    # DE overlap follows the pinned scorer; no official hidden values enter this audit.
    overlap,details=core._signed_overlap(pred_delta,up,down)
    report={'created_utc':now(),'scope':'Observed E8.5/E9.5 and saved models only; no hidden E10.5 expression or score-derived reconstruction.',
        'official_result_sha256':digest(result_path),
        'source_report_sha256':digest(HERE/'private/state_search_01/report.json'),
        'fit_support_shift':{'local_E8_5_trusted_fraction':local['trusted_donor_fraction'],
            'final_E9_5_trusted_fraction':official_audit['trusted_donor_fraction'],
            'local_state_support':local['state_support'],'final_state_support':official_audit['state_support'],
            'interpretation':'Fewer anchor cells satisfy the same source-distance guard at final cutoff. Supports domain/support mismatch as a concern, not proof of why hidden score fell.'},
        'local_DE_diagnostic':{'truth_de_count':len(truth_de),'protected_truth_de_count':int(np.isin(truth_de,protected).sum()),
            'protected_gene_count':len(protected),'signed_overlap':float(overlap),
            'predicted_vs_true_delta_std_ratio':float(pred_delta.std()/true_delta.std()),
            'full_panel_delta_spearman':float(spearmanr(pred_delta,true_delta).statistic)},
        'official_skill_minus_50':{k:v-50 for k,v in result['metric_skills_percent'].items()},
        'remaining_uncertainties':['Official hidden target raw metrics and expression are unavailable.',
            'One local later stage does not establish that earlier temporal changes persist into E10.5.',
            'Cell-level resampling cannot measure uncertainty from independent embryos.',
            'Protected-gene limitations, state coverage and temporal extrapolation need separate mechanism tests.'],
        'next_action':'Require a same-horizon, tissue-matched past-only validation test and diagnose biological trajectory transfer before another prospective export. Keep official quota unused.',
        'no_training_performed':True}
    (HERE/'OFFICIAL_TRANSFER_FAILURE.json').write_text(json.dumps(report,indent=2))
    (HERE/'OFFICIAL_TRANSFER_FAILURE.md').write_text(
        '# Confirmed official transfer failure\n\n'
        'The user confirmed that T1_val__unit16_progress_20260928_01.h5ad scored 46.59, rank 217/268. '
        'Component skills: DE 38.2, direction 51.3, MMD 48.6, variogram 48.2. Three metrics fall below persistence. '
        'The earlier local mean 55.31 was not an official estimate; the progress candidate fails the official task. '
        'Retain unit_k16 as a local comparison only, not as a deployable or promoted model.\n\n'
        'The saved source-distance guard accepted 83.3% of E8.5 donor cells locally, falling to 38.8% at the final E9.5 fit. '
        'This is evidence of changing support under the selected representation. It does not isolate the cause of hidden E10.5 error. '
        'The audit JSON records state support, known-stage DE coverage and delta alignment without accessing E10.5 ground truth.\n\n'
        'Next work must test same-horizon tissue transfer, gene-program direction and cell-state coverage as separate mechanisms. '
        'Quarter-day atlas checks and cell bootstrap stability cannot establish one-day transfer to another heart stage. '
        'Do not resubmit this model unchanged, convert 55.31 into an official prediction, tune hidden genes from four scores, '
        'or waive the existing >72 promotion gate. The previous export was explicitly user-requested progress.\n')
    state_path=HERE/'LOCAL_OPTIMIZATION_STATE.json'
    state=json.loads(state_path.read_text())
    state['official_transfer_failure']={'score':46.59,'candidate':'unit16_progress_20260928_01',
        'report':'outputs/t1_iterations/OFFICIAL_TRANSFER_FAILURE.json',
        'report_sha256':digest(HERE/'OFFICIAL_TRANSFER_FAILURE.json'),'status':'rejected_for_official_use'}
    state['candidate_role']='Local control only; official progress export failed'
    state['updated_utc']=now();state['local_process_running']=False
    state['next_experiment']=report['next_action']
    state['progress_export'].update(uploaded=True,official_score=46.59,
        result_source='User screenshot and explicit confirmation',official_result_sha256=digest(result_path))
    state_path.write_text(json.dumps(state,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
