"""Checkpoint completed horizon audits and publish complete metric vectors."""
import json
import os
import subprocess
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now


def main():
    runs={};running=[];unfinished=[];process_probe_error=None
    commands=[]
    if os.name=='nt':
        try:
            probe=subprocess.run(['powershell','-NoProfile','-Command',"Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' } | Select-Object -ExpandProperty CommandLine | ConvertTo-Json -Compress"],capture_output=True,text=True,timeout=15,check=True)
            parsed=json.loads(probe.stdout or '[]');commands=[parsed] if isinstance(parsed,str) else parsed
        except (OSError,ValueError,subprocess.SubprocessError) as exc:process_probe_error=type(exc).__name__
    else:process_probe_error='Process probe unavailable on this platform'

    for name in ['matched_horizon_audit_01','quantile_horizon_pilot_01','state_quantile_horizon_pilot_01',
                 'annotation_horizon_pilot_01','annotation_support_pilot_01',
                 'program_horizon_pilot_01','cell_program_horizon_pilot_01','cell_program_repair_01','cell_program_feature_pilot_01','empirical_bayes_horizon_01','covariance_horizon_01','empirical_bayes_repair_01','copula_horizon_01','neural_ode_horizon_01','neural_hurdle_horizon_01','neural_sampling_horizon_01','growth_composition_horizon_01','transport_hurdle_horizon_01','transport_decoder_horizon_01','transport_conditional_horizon_01','stage_transport_horizon_01','resolution_transport_horizon_01','dimensional_transport_horizon_01']:
        folder=HERE/'private'/name;path=folder/'report.json'
        if not path.exists():
            if folder.exists():
                unfinished.append(name)
                planpath=folder/'plan.json'
                if planpath.exists():
                    planned=json.loads(planpath.read_text())
                    main_script=next(iter(planned.get('source_sha256',{})),None)
                    if main_script and any(main_script in (command or '') for command in commands):running.append(name)
            continue
        r=json.loads(path.read_text());summaries=[]
        for candidate in r['plan']['configs']:
            results=[next(v for v in f['results'] if v['candidate']==candidate) for f in r['folds']]
            valid=all(v['calibration_valid'] for v in results)
            summaries.append({'candidate':candidate,'all_calibrations_valid':valid,
                'scores':[v['local_score'] for v in results],
                'mean_score':sum(v['local_score'] for v in results)/len(results) if valid else None,
                'mean_skills_percent':{k:100*sum(v['skills'][k] for v in results)/len(results)
                    for k in results[0]['skills']} if valid else None,
                'beats_persistence_on_every_fold':valid and all(v['local_score']>50 for v in results),
                'all_metrics_at_least_50_on_every_fold':valid and all(min(v['skills'].values())>=.5 for v in results)})
        runs[name]={'report_sha256':digest(path),'plan_sha256':digest(folder/'plan.json'),
            'folds':r['folds'],'summaries':summaries,'scope':r['plan']['scope']}
    status='running' if running else 'incomplete' if unfinished else 'completed'
    output={'updated_utc':now(),'status':status,'runs':runs,'running':running,'unfinished':unfinished,'process_probe_error':process_probe_error,
        'local_72_gate_passed':False,'official_score_of_new_model':None,'submissions_used':0,'jev_requests_used':0}
    (HERE/'HORIZON_BATCH_RESULTS.json').write_text(json.dumps(output,indent=2))
    lines=['# One-day source-cohort audits','',
        'Two observed one-day folds, E8.0→E9.0 and E8.25→E9.25, use the same cardiac-associated source cohort, '
        '1,500 donors and complete official gene ordering. Each forecast is frozen before later-stage expression is read for scoring. '
        'Missing/ambiguous genes use zero placeholders in this atlas-only diagnostic; these are not measured challenge genes. '
        'These folds therefore improve horizon matching but do not certify challenge-domain or embryo-independent generalization.','',
        'The prior model scores 47.143 and 47.987. All component ablations also score below persistence on both folds. '
        'This confirms one-day failure within the source cohort; it cannot isolate hidden E10.5 causes.','',
        'The positive-quantile mechanism fits three historical positive-expression quantile distributions, suppresses reversing trends, '
        'shrinks low-support slopes, and preserves the donor zero mask and number of cells. Per-cell mapped mass and protected genes are guarded. '
        'Marginal proposals are monotone; cell-specific mass conservation can change final cross-cell ranks.','',
        'A second pilot conditions those margins on four broad past-fitted states. It retains fixed donor counts, '
        'changes only trusted anchors, requires 20 anchors and 20 source cells at every recent stage, '
        'and applies both within-state and overall covariance guards. It tests composition confounding, not inferred cell proliferation.','',
        'The annotation pilots group cells by observed published type strings and estimate positive-abundance and detection trends separately. '
        'Four ablations compare expression, detection and their combination. A second frozen pilot raises the source support requirement '
        'from 20 to 100 cells per type per past stage and the positive-expression minimum from10 to20. Donor counts stay fixed. '
        'Annotations now enter learner grouping; joint atlas annotation is a limitation, and later-stage label values are excluded from fitting.','',
        'Shared temporal-profile pilots project gene slopes onto past-only SVD ranks2/4/8 with detection0/.5. '
        'These orthogonal factors are denoising statistics, not biologically validated gene programs. '
        'A separate cell-level pilot uses NMF ranks8/16, three fixed replicas, 3000 past fit cells, '
        '384 past-selected genes, scaled normalized abundance, median consensus and fixed-component usage fitting. '
        'A ridge1 full-panel decoder transfers within-type usage slopes to bounded gene factors. '
        'Convergence warnings and consensus dispersion are retained in every model audit.','',
        'The initial16-factor cell-level fit hit the200-iteration budget. A separate frozen optimizer repair '
        'raises only that budget to800 for rank16, retaining source cells, genes, random seeds, tolerance, '
        'decoder and evaluator panels. Higher iteration count is not assumed to improve forecast accuracy.','',
        'The2048-gene coverage ablation converged but regressed. Empirical Bayes pilots tested normal-mixture slope shrinkage with error inflation1/2 and sign gates none/.1/.25. Initial EM fits hit1000 iterations; a separate100-step EM plus convex-simplex SLSQP repair converged with scientific settings unchanged. Neither produced meaningful forecast gains.','',
        'Covariance pilots used8 past-fitted PCA dimensions and compared OAS with empirical covariance at three strengths. Positive-margin rank coupling used those covariance proposals, fixed zero masks and exact per-type margins before factor clipping and per-cell mass conservation. Projection changes final margins; all actual metrics remain required. Neither branch meaningfully exceeded persistence.','',
        'The neural ODE pilot uses a Gaussian latent encoder, reconstruction pretraining and Sinkhorn losses for observed distributions and latent dynamics. Two fixed regularization values and three forecast strengths use all permitted past cells in the prepared cohort as minibatch pools. This is not the complete raw atlas, and stochastic batches do not guarantee every cell was sampled. A past-only full-gene ridge decoder transfers conditional latent-mean drift to bounded donor-relative factors. No type labels enter the neural learner. The final training iteration is fixed before target scoring. Complete forecast metrics, rather than training loss, determine progress.','',
        'Author implementation: [scNODE](https://github.com/rsinghlab/scNODE), architecture, solver, training, loss and benchmark sources reviewed and pinned in NEURAL_ODE_AUTHOR_REFERENCE.json. The full paper fetch was blocked. Normalization, velocity and gradient bounds, ridge decoding and conditional-mean forecasting make this an adaptation rather than a reproduction.','',
        'A subsequent ablation fits separate detection and positive-abundance heads on frozen beta0 neural latent means, with abundance-only, detection-only and joint forecasts at two strengths. It permits zero-mask changes with common fixed random uniforms, conserves mapped mass and retains protected genes. Clipped linear probability estimates and diagonal-shrunken conditional abundance coefficients are heuristic adaptations. No additional neural dynamics training or target-selected checkpoint is used. These are reused development folds, not fresh blind validation.','',
        'A numerical sampling ablation keeps those fitted heads exactly unchanged, verifying every recomputed head array against its archive. Random-order systematic Bernoulli draws constrain each gene\'s switch count to within one of its expectation before final projection. Within-gene switches become dependent; improved scores or biological covariance are not guaranteed. Independent-draw forecasts are replayed with verified hashes as matched controls. The frozen plan\'s abundance-only-control phrase refers to the preceding head pilot; the sampling trial directly compares independent and systematic detection-only/joint forecasts.','',
        'The growth-proxy branch pins the author mouse marker files and fits historical expression-matched scores. All94 proliferation and187 P53-proxy genes map uniquely. The first audit failed because its expressed-only control filter left insufficient controls for low-expression markers. A separate repair retains zero-expression controls and preserves that failure. The preliminary forecast test resamples complete donor cells using bounded proliferation, negative-P53 or net weights at two strengths, with exact neutral-growth identity and unchanged calibration. This tests composition sensitivity, not an unbalanced transport solver or measured embryonic birth/death rates. Source: [moscot marker implementation](https://github.com/theislab/moscot/blob/440093ccbb8e70de209157d91da839c55b897821/src/moscot/utils/data.py) identifies its mouse apoptosis list as a P53-pathway proxy; [author marginal tutorial](https://moscot.readthedocs.io/en/stable/notebooks/examples/problems/TemporalProblem/800_score_genes_for_marginals.html) discusses prior-growth sensitivity.','',
        'Sources: [Stephens supplement](https://stephenslab.uchicago.edu/assets/papers/Stephens2017-supplement.pdf), sectionsS.1-S.2; [Chen et al.](https://arxiv.org/html/0907.4698v1), Gaussian assumptions, OAS derivationIII-C and simulationIV; [sklearn covariance documentation](https://scikit-learn.org/stable/modules/covariance.html). These motivate estimator adaptations and do not establish developmental forecasting accuracy.','',
        'Literature: [Schefzik, Thorarinsdottir and Gneiting (2013)](https://arxiv.org/html/1302.7149v2), '
        'methods 4.1–4.3 and experiments 5.4 read. ECC separates marginal calibration from rank dependence. '
        'Its weather experiments found benefits dependent on the dependence structure. This implementation adapts that separation '
        'to historical positive scRNA margins; it is neither faithful ECC nor evidence that ECC improves this challenge.','',
        'The [muscat paper](https://www.nature.com/articles/s41467-020-19894-4) motivates separating within-subpopulation state changes '
        'from differential abundance. Introduction, simulation results (20–400 cells) and simulation-preprocessing methods were read. '
        'It found sizable detection gains between20 and100 cells per subpopulation/sample. It uses replicated samples and count-based '
        'inference; our normalized-abundance trend forecast does not reproduce those methods, biological replication or their inferential guarantees.','',
        '[Kotliar et al. (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6639075/) distinguishes identity and activity programs '
        'and warns that type averages can miss activity and statistical factors need not be biological programs. '
        'Introduction, simulation benchmark, preprocessing and consensus methods were read. '
        'The cell-level adaptation uses fewer genes/replicas, normalized abundance, no component outlier filtering, '
        'and added ridge decoding and temporal extrapolation; it is not cNMF reproduction or a test of the paper\'s biological claims.','',
        'Historical transport fits past-only PCA representations and quarter-day couplings, then extrapolates barycentric velocities with a ridge affine field. Balanced transport and three unbalanced prior settings use the same frozen full-panel scoring. All first-pilot candidates fail persistence on both folds. The entropy convention is explicit and differs from newer POT KL-reference defaults; this is an adaptation, not WOT/moscot reproduction. Couplings use256 cells per stage and PCA uses3000 fit cells, so these are bounded computational pilots, not full raw-atlas transport.', '',
        'A frozen-flow decoder ablation verifies every archived head array, then separates positive abundance, detection switches and systematic joint sampling. Abundance-only improves over joint forecasts on both development folds, but its best mean50.0276 combines51.7018 and48.3534 and fails consistency. This motivates a declared full conditional latent covariance ridge ablation; model fitting remains past-only. All folds are exposed development data, not untouched validation.', '',
        '| Run | Candidate | Fold 1 | Fold 2 | Mean |','| --- | --- | ---: | ---: | ---: |']
    for name,r in runs.items():
        for s in r['summaries']:
            if s['mean_score'] is not None:
                a,b=s['scores'];lines.append(f"| {name} | {s['candidate']} | {a:.3f} | {b:.3f} | {s['mean_score']:.3f} |")
    lines += ['',f'Status: {status}. Full four-metric vectors, raw metrics, model audits and provenance hashes are in HORIZON_BATCH_RESULTS.json.',
        'No new prospective export or official submission. The >72 objective remains unfinished.']
    (HERE/'HORIZON_BATCH_RESULTS.md').write_text('\n'.join(lines)+'\n')
    path=HERE/'LOCAL_OPTIMIZATION_STATE.json';state=json.loads(path.read_text())
    state.update(updated_utc=now(),local_process_running=bool(running))
    state['horizon_batch']={'status':status,'running':running,'report':'outputs/t1_iterations/HORIZON_BATCH_RESULTS.json',
        'report_sha256':digest(HERE/'HORIZON_BATCH_RESULTS.json')}
    if running:state['next_experiment']='Finish frozen batches: '+', '.join(running)+'. Preserve paired controls and all four metrics.'
    path.write_text(json.dumps(state,indent=2))
    path=HERE/'METRIC_RESEARCH_QUEUE.json';queue=json.loads(path.read_text())
    entry={'id':'positive_quantile_temporal','metrics':['de_score','de_direction','mmd_u','variogram'],
        'status':'running' if any('quantile' in r for r in running) else 'implemented_evaluated',
        'sources':[{'url':'https://arxiv.org/html/1302.7149v2',
            'read':'Methods sections4.1-4.3 and experiment section5.4; marginal/rank separation and limits.'}],
        'implementation':['positive_quantile_forecast.py','state_quantile_forecast.py'],
        'scope':'Adaptation, not faithful ECC. Three positive-quantile strengths, global and four-state conditional, two one-day associated-cohort folds.',
        'results_report':'outputs/t1_iterations/HORIZON_BATCH_RESULTS.json',
        'results_report_sha256':digest(HERE/'HORIZON_BATCH_RESULTS.json'),
        'research_family_exhausted':False,'submissions_used':0,'jev_requests_used':0,
        'next_action':state['next_experiment']}
    for value in queue.values():
        if isinstance(value,list) and any(isinstance(v,dict) and 'id' in v for v in value):
            previous=next((i for i,v in enumerate(value) if v.get('id')==entry['id']),None)
            if previous is None:value.append(entry)
            else:value[previous]=entry
            annotated={'id':'annotation_conditioned_temporal',
                'metrics':['de_score','de_direction','mmd_u','variogram'],
                'status':'running' if 'annotation_support_pilot_01' in running else 'implemented_evaluated',
                'sources':[{'url':'https://www.nature.com/articles/s41467-020-19894-4',
                    'read':'Introduction, simulation results20-400 cells and preprocessing methods; separates differential state and abundance.'}],
                'implementation':['annotation_trend.py','annotation_horizon_pilot.py','annotation_support_pilot.py'],
                'scope':'Heuristic adaptation to normalized abundance; not muscat replication. Observed annotations group past cells. Four expression/detection ablations, source support20 vs100, two one-day folds.',
                'results_report':'outputs/t1_iterations/HORIZON_BATCH_RESULTS.json',
                'results_report_sha256':digest(HERE/'HORIZON_BATCH_RESULTS.json'),
                'limitations':['Published annotation may derive from joint stages; retrospective diagnostic only.',
                    'Capture IDs do not establish independent embryos; cell-based errors are shrinkage heuristics.',
                    'Missing official genes are atlas zero placeholders, not measured zeros.'],
                'research_family_exhausted':False,'submissions_used':0,'jev_requests_used':0,
                'next_action':state['next_experiment']}
            previous=next((i for i,v in enumerate(value) if v.get('id')==annotated['id']),None)
            if previous is None:value.append(annotated)
            else:value[previous]=annotated
            programs={'id':'shared_gene_program_dynamics','metrics':['de_score','de_direction','mmd_u','variogram'],
                'status':'running' if any('program' in r for r in running) else 'implemented_evaluated',
                'sources':[{'url':'https://pmc.ncbi.nlm.nih.gov/articles/PMC6639075/',
                    'read':'Introduction, simulation benchmarks, count/TPM variance scaling, consensus methods and limitations.'}],
                'implementation':['program_trend.py','cell_program_trend.py','program_horizon_pilot.py',
                    'cell_program_horizon_pilot.py','cell_program_repair.py'],
                'scope':'Shared SVD profile directions versus cell-level NMF median consensus usage dynamics. Neither is a faithful cNMF reproduction or certified biological-program recovery.',
                'frozen_trials':'SVD ranks2/4/8 with detection0/.5; NMF ranks8/16 with detection0/.5,3 fixed seeds and200 iterations. Rank16 optimizer repair800 iterations. Two matched one-day development folds.',
                'results_report':'outputs/t1_iterations/HORIZON_BATCH_RESULTS.json',
                'results_report_sha256':digest(HERE/'HORIZON_BATCH_RESULTS.json'),
                'research_family_exhausted':False,'submissions_used':0,'jev_requests_used':0,
                'next_action':state['next_experiment']}
            previous=next((i for i,v in enumerate(value) if v.get('id')==programs['id']),None)
            if previous is None:value.append(programs)
            else:value[previous]=programs
            break
    path.write_text(json.dumps(queue,indent=2))
    print(json.dumps({'status':status,'running':running,'means':{
        k:{s['candidate']:s['mean_score'] for s in v['summaries']} for k,v in runs.items()}},indent=2))


if __name__=='__main__':main()
