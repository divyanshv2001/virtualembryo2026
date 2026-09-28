# Offline evaluation and submission conservation

The user's current instruction is to preserve the four remaining daily submissions and iterate locally before another upload. This supersedes earlier recommendations to submit a persistence control or another exploratory candidate. No submissions or Jev requests were used in this work.

An official T1 score above 72 cannot be certified locally while the E10.5 target is hidden. The [official rules](https://virtualembryo.ai/challenge/rules), section 10, state that ranking targets are not distributed. The [public local scorer](https://github.com/aristoteleo/veckit) also explicitly distinguishes a supplied pseudo target from the real competition score. Do not relabel a reconstruction score, an earlier-stage backtest, or a score using official anchors with a different target as a leaderboard forecast.

## What was executed

- Downloaded only public scorer code and its licence, pinned to `46d41e63f42a9aab815db20b742feeccd249cb17`; biological samples bundled in that repository were not fetched.
- Audited the source. `common/core_metrics.py` implements the current DE null correction, partial-rank direction, target-only PCA/unbiased multiscale MMD, and gene-pair variogram. `T1/metrics.py` contains older metrics; it was inspected and is not used.
- Downloaded the author-released WT atlas from [Cambridge](https://gastrulation.stemcells.cam.ac.uk/), linked by [Scialdone et al.](https://doi.org/10.1038/nature18633). Its 1,205 cells cover E6.5, E7.0, E7.5 and E7.75 only. The separate Tal1 experiment was not downloaded. This is T1 early-stage research, not T2 training. No competition prediction uses this atlas, and the data are not redistributed. Any future competition use needs the source/use disclosure and applicable data terms checked separately.
- Parsed the gzip file as a tar archive and read only the named counts member without extracting filesystem paths. Checked cell/metadata identities and stage counts. Normalized each cell's counts to 10,000 and applied natural log1p. The local feature panel has 41,388 Ensembl IDs, not the challenge's 32,285-symbol panel; no gene mapping or challenge-score parity is asserted.
- Froze ten models before evaluation: persistence, global mean drift, gene-quantile drift and local-neighborhood mean drift, with strengths 0.25, 0.5 and 1.0 for the three drift families. Fit E6.5/E7.0 and forecast E7.5. Three fixed embryo splits assess sensitivity. Selection never supplies target expression or target-fitted PCA to the prediction function.
- Evaluated 30 candidate/panel combinations locally. The initial run was followed by two technical corrections, not three independent experimental studies. Original event logs and executed source/report history remain under ignored `private/`.

## Self-corrections and final outcome

The initial E7.75 test reported 60 for persistence, which is invalid: its variogram split-embryo ceiling was worse than its copy-last floor. The upstream hyperbolic transform assumes correctly ordered anchors; finite outputs alone are insufficient to validate them. Added a guard requiring the ceiling to outperform the floor for every metric before producing an aggregate. Invalid calibration now returns no score, instead of dropping a metric or giving it spurious credit.

One E7.5 cell had no embryo identifier. The first implementation converted it to a string and treated it as a group. Corrected the grouping to reject unknown IDs and explicitly exclude that cell from embryo-level validation. The original later-stage test has already been read; corrected reruns are not fresh blind tests. No new models or strengths were introduced after reading it.

In the corrected run, two of three E7.5 panels calibrated successfully; the third was rejected. Across the two valid panels, persistence scored 50.0 and every drift model scored below 50. Their means ranged from 29.6 to 42.8. These partial-panel averages are diagnostics, not an accepted three-panel result or a leaderboard estimate. The required three-panel gate failed, so no candidate was selected or promoted and no local >72 result was attained. The invalid initial 60 is superseded and must not be used as evidence.

The completed report is `private/offline_backtest_03_calibration_guard/report.json`. Its status is `calibration_failed`, with `local_threshold_reached=false`, `official_72_verified=false` and `submissions_used=0`. The initial failed correction preserved its failure log instead of being silently overwritten.

## What this loop can and cannot establish

This atlas is small and assays FACS-selected early mesoderm/epiblast with SMART-seq; the competition supplies later-stage heart-region RNA. Its changing sample-selection regime and heterogeneous embryo groups make it a weak proxy, and its rejected calibration prevents using an aggregate threshold for model selection. Changing calibration seeds or dropping unfavorable metrics until a score exceeds 72 would not solve that limitation.

A credible next forecast backtest needs a larger permitted, compatible time series with at least three usable stages and independent sample metadata. Freeze source stages, preprocessing, development targets, a final holdout, models, losses and stopping criteria before evaluation. Select on development results, evaluate the final holdout once, and retain uncertainty and negative results. An earlier-stage result exceeding 72 would still be a proxy result, not proof of a hidden E10.5 score. Official anchors are not used to reconstruct target values or turn external-atlas scores into leaderboard predictions.

Keep all four remaining attempts unused until a candidate has defensible local support. The existing 48.1 full-shift file remains the best measured leaderboard artifact; no local run has established a superior submission.

## Reproduce

```powershell
outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/fetch_local_scorer.py
outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/fetch_early_atlas.py
outputs/research_workflow/.venv/Scripts/python.exe -m unittest discover -s outputs/t1_iterations -p test_offline_backtest.py
outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/offline_backtest.py --round NEW_UNUSED_ROUND_NAME
```

The download steps need network access; the model/evaluation script is offline and has no submission or API capability. The requirements snapshot is `requirements-local-lock.txt`. Tests cover cached DE/direction/MMD agreement with the pinned core on independent synthetic inputs, intact embryo grouping, a zero-horizon identity check, and rejection of an incorrectly ordered ceiling. Variogram formulas match the published definition, but the wrapper caches a fixed pair draw and batches its calculation; sampling seeds and array orders need not match the hosted evaluator. No exact hosted-scorer parity claim is made. Licence and source attribution for the public metric code are retained in `SCORER_LICENSE.txt` and the private source manifest.
