# Weak-fold diagnosis and robust population forecasts

Implemented the next development branch and completed **96 additional model/fold evaluations**: eight fixed configurations on six rolling splits in two cohorts. The 72 objective remains unmet. No candidate passed promotion, and no challenge file was exported or submitted. No Jev request was made.

## What failed in the earlier weak fold

`diagnose_weak_fold.py` reconstructed the previously evaluated global population tilt 0.5 / expression trend 0.1, fitting only stages through E8.25 before inspecting E8.5. Its aggregate deficit relative to 50 decomposed into DE recovery −0.337 points, direction +0.058, MMD +0.281 and variogram −0.068. Annotated population total variation remained 0.265 for both persistence and the candidate. This diagnostic uses all target cells and annotations; it is descriptive development error analysis, not a new blind metric panel.

Mean-expression error worsened within several sufficiently sampled annotation groups, including Epicardium (RMSE 0.07457→0.07770), Cardiomyocytes FHF 2 (0.06473→0.06742), and anterior cardiopharyngeal progenitors (0.06741→0.06960). These label-conditioned comparisons do not establish ancestry or a cause of the official leaderboard deficit.

The old 24-state model had sufficient three-stage support for only one state at the E8.0 cutoff: 23 states had fewer than eight cells at at least one recent stage. This explains why its conditional expression component frequently fell back to unchanged donors. It does not prove that coarser states forecast better.

## Expanded cohort and model changes

`prepare_extended_atlas.py --domain cardiac_associated --cells-per-stage 3000` prepares a fixed 28-annotation cohort: the existing cardiac/endocardial/epicardial groups plus explicitly declared associated mesenchymal, pharyngeal, neural-crest, gut, endothelial, ectodermal and blood populations. The labels restrict the cohort only; they are not model inputs. No target-expression similarity chooses labels or cells. Natural within-stage population proportions are retained by uniform sampling rather than reweighted to match a hidden target.

The resulting `private/associated_prepared_01` contains **25,963 × 27,669** values. It retains 1,963 cells at E7.5 and samples 3,000 at each subsequent quarter-day stage through E9.5. Unknown/mixed stages are excluded. The source checksums and integer-count gates pass before full-panel log1p library normalization to 10,000. Expression SHA-256: `ef68dc70b341eb2bbbae9610fc88132a116c4526f334d1d6d013849d47a9c387`. The cohort remains an approximation: several challenge labels have no exact atlas equivalent, and published annotations may have used joint atlas analysis.

`robust_population.py` learns four or eight training-only k-means states rather than 24. It suppresses trends that reverse sign over the two recent training intervals, shrinks by mean-estimation uncertainty, and applies additional small-state shrinkage. Conditional abundance statistics are computed in gene blocks. Feature selection, PCA, clustering, state proportions and abundance trends all exclude the fold's future stages.

Donor reweighting requires effective sample size ≥80% of donors. Population and expression changes back off until covariance changes by at most 10% in Frobenius norm across all 384 training-selected features, relative to the original reference donors. No target values enter this backoff. Expression factors are capped at [0.8, 1.25]; donor zero patterns are retained. These are engineering constraints and hypotheses, not guarantees of future accuracy. Unreliable expression states retain donor expression.

## Completed results

Both the original narrow cardiac cohort and the expanded associated cohort were evaluated on E8.0→E8.25, E8.0→E9.0, E8.25→E8.5, E8.5→E8.75, E8.5→E9.5 and E8.75→E9.0. Each fold fits all available sampled past cells, then uses a fixed subset of up to 1,000 reference donors and 1,000 target cells for evaluation. Target cells are split into random halves for a sampling ceiling. This is not independent embryo replication.

All stages are development: E8.5 and E9.5 have already been inspected. Sampling and calibration differ from the earlier four-split run, so its 68.07 and the values below are not identical-panel comparisons.

| Cohort / configuration | Mean across six splits | Worst split | Best split |
|---|---:|---:|---:|
| Persistence, either cohort | 50.00 | 50.00 | 50.00 |
| Narrow cardiac, incumbent global | **55.49** | 49.65 | 64.43 |
| Narrow cardiac, robust 8-state / expression 0.25 | 50.79 | 49.99 | 53.67 |
| Associated, incumbent global | 53.08 | 47.81 | 63.65 |
| Associated, robust 8-state / expression 0.5 | 52.92 | 44.73 | 61.19 |

Every metric panel passed calibration, but **no configuration passed the promotion gate**: aggregate >50 and each metric skill ≥50 on every split, plus mean >72. The stronger constraints did not outperform the incumbent across folds. Enlarging the cohort improved early support (all four coarse states supported at E8.0), but some later windows still lacked state overlap. The expanded populations did not produce consistent score gains. Preserve these negative results; do not choose only favorable stages or relax calibration to claim success.

The 84 guarded forecasts, including persistence controls, had maximum measured covariance change **0.09967**, no newly positive donor-zero entries, and effective donor sample sizes within the declared bounds. Meeting those constraints did not ensure a better distribution or variogram against future observations.

## Transfer implementation and checks

`transfer_genes.py` maps only unique exact symbols. Missing and ambiguous atlas genes retain their float32 challenge donor values exactly. Abundance mass is conserved within mapped genes, so subsequent full-panel renormalization cannot silently alter protected genes. Invalid factors, duplicate official symbols and mismatched shapes are rejected. The helper does not select a candidate, export a submission or bypass promotion. Full challenge transfer and file validation remain conditional on a passing candidate.

Ten relevant tests passed: robust trend reversal/uncertainty behavior, future-data invariance, effective sample size, covariance bounds, persistence, support preservation, binary decoding, normalization and protected-gene mapping/mass conservation. These verify implementation properties, not a score of 72.

```powershell
outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/diagnose_weak_fold.py
outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/prepare_extended_atlas.py --round NEW_ASSOCIATED_DATA --domain cardiac_associated --cells-per-stage 3000
outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/robust_backtest.py --dataset cardiac_prepared_01 --round NEW_NARROW_RUN
outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/robust_backtest.py --dataset NEW_ASSOCIATED_DATA --round NEW_ASSOCIATED_RUN
outputs/research_workflow/.venv/Scripts/python.exe -m unittest discover -s outputs/t1_iterations -p test_robust_population.py
outputs/research_workflow/.venv/Scripts/python.exe -m unittest discover -s outputs/t1_iterations -p test_transfer_genes.py
```

Actual executed sources, frozen plans, checkpoints, donor selections and event/metric logs are retained under ignored `private/weak_fold_audit_01`, `private/associated_prepared_01`, `private/robust_cardiac_01` and `private/robust_associated_01`. Existing rounds cannot be overwritten. This adds 96 evaluations to the prior 126; none certifies the official hidden E10.5 leaderboard score. The four remaining user-reported submission slots remain unused by this work.
