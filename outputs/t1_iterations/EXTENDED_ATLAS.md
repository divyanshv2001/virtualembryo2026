# Extended mouse atlas training

This workflow implements the user's request to acquire the larger atlas and start local training while preserving the four remaining daily submissions. Downloads, training and evaluation have no upload capability and use no Jev requests.

## Source and acquisition

The old `rosshandler/EmbryoTimecourse2020` README points to a stale cloud share. The current source is the [author portal](https://marionilab.github.io/ExtendedMouseAtlas/) linked by [Imaz-Rosshandler et al., Development 2024](https://doi.org/10.1242/dev.201867). The publication states CC BY 4.0 and links the processed atlas and raw accessions. The author portal links the [UCSC mirror](https://cells-test.gi.ucsc.edu/?ds=ext-mouse-atlas).

The Cambridge download bundles all formats into a 25 GB archive. Acquired the author-linked UCSC metadata, expression index and 1,564,242,043-byte expression binary instead. The binary SHA256 is `583c09465d844a0940b6461a49806cdeba3b291a56809e7ee9cbc74188bc163a`. All downloaded files, their URLs, ETags and SHA256 hashes are recorded in `private/extended_atlas/manifest.json`. No credentials are needed. Raw biological data remain local and ignored by Git.

Cell Browser's [public format implementation](https://github.com/maximilianh/cellBrowser/blob/master/src/cbPyLib/cellbrowser/cellbrowser.py) documents separately zlib-compressed gene vectors, a two-byte description length and a four-byte expression value per cell. The wrapper decodes that format directly, validates the metadata length and reads the source's Float32 values. The index has 27,669 genes and one `_range` metadata entry; the first strict preparation attempt rejected that extra entry. The corrected preparation explicitly accepts only `_range=[0,0]` and preserves the initial failure directory.

## Stage and preprocessing gates

The metadata contain 430,339 cells. Explicit stages range from E6.5 through E9.5 in quarter-day increments; 7,455 cells labelled `Mixed gastrulation` have no explicit stage and are excluded. Do not infer their stages from their expression or allow them into training. The script rejects any unexpected numeric stage. The permitted explicit range is consistent with T1's [external-data rules](https://virtualembryo.ai/challenge/rules); this pipeline is not a T2 data preparation workflow.

Predeclare a uniform seeded sample of 1,000 cells per explicit stage: 13,000 cells, retaining every indexed gene. Require nonnegative, finite, integer-valued counts in selected cells. Normalize each cell across the complete atlas panel to a total count of 10,000, then apply natural log1p. Process the source one gene at a time and store the staged sample in a memory-mapped NumPy file to bound memory use. The provided UMAP, batch-corrected PCA, cell clusters and cell-type labels are not learner inputs.

`sample` is a capture/sample identifier. `embryo_version` means Original versus Extended atlas, not an individual embryo. These fields cannot support an invented independent-embryo confidence interval. Local ceiling calibration uses disjoint random cell halves and is described as sampling replication only.

## Training and selection

Freeze the plan before fitting. Development models train only on E6.5–E7.5 and forecast E8.5, one day beyond the last fitted stage. Select by the development metric panel only. Refit the selected configuration on stages through E8.5, then evaluate E9.5 once. The same later stage cannot serve as both fit data and forecast validation. Reading the complete source to prepare stage-separated arrays does not fit a representation on later stages: feature variances, scalers, PCA, decoder and dynamics use explicitly restricted training rows.

Nine declared configurations are compared: persistence; half/full mean trends fitted on the three latest training stages; and half/full latent velocity rollouts with linear ridge penalties 10 and 100 or a 128-feature random-Fourier expansion with ridge penalty 100. These are hypotheses, not claimed winners.

## Completed first training batch

Preparation completed with a 13,000 × 27,669 log-normalized staged sample. The downloaded count-valued source passed the count gate. Development representation/decoder used 5,000 cells through E7.5; the three distinct velocity regressors trained on 4,000 adjacent-stage pseudo-pairs. All nine forecast configurations were evaluated after forecasts were frozen. The half-strength linear velocity model with ridge penalty 10 had the highest development score, 54.22, against persistence at 50. Its in-sample pseudo-pair RMSE was 2.04; that is a training-fit diagnostic, not forecast accuracy.

The selected configuration was refitted through E8.5 on 9,000 cells and 8,000 pseudo-pairs. Its once-used E9.5 backtest score was **34.98**, with valid local calibration. Both stages' scores are atlas-local proxy values, not competition results. The >72 development/final gate did not pass. No submission was generated or recommended from this result.

| Selected model | DE recovery skill | Direction skill | MMD skill | Variogram skill | Local total |
|---|---:|---:|---:|---:|---:|
| E8.5 development forecast | 52.08 | 57.26 | 58.95 | 46.00 | 54.22 |
| E9.5 once-used forecast test | 64.65 | 58.06 | 9.78 | 6.86 | 34.98 |

The final test improved DE/direction relative to its local floor but damaged population distribution and variograms enough to dominate the aggregate. This falsifies promotion of this fitted rollout as a strong model. It does not uniquely identify the cause: approximate state matches, velocity extrapolation, clipping, support changes, normalization and atlas sampling all differ from a correct future distribution. A further development branch should constrain those distribution changes and use rolling forecast checks; the already-read E9.5 result is not a fresh blind test for that branch.

The symbol coverage audit found 26,796 shared symbols with the official 32,285-gene panel, leaving 5,489 official genes absent from this atlas panel. There are also 26 duplicate atlas symbols. Do not silently drop the missing genes or merge duplicated symbols without an explicit mapping policy. Transfer to the challenge input data requires a separate fitted mapping/reconstruction step; these checkpoints are not drop-in challenge predictions.

The velocity models fit a 24-dimensional whitened PCA using 768 training-only high-variance genes. Adjacent training-stage cells receive 16-neighbor earlier-state barycentres, creating approximate velocity targets. Ridge regression learns a state/time-dependent velocity. These pseudo-pairs are neither known lineage nor optimal-transport couplings. Rollouts step by 0.25 days and cap speed using a training-only 95th-percentile bound. A ridge decoder projects latent displacement into all atlas genes while retaining donor expression residuals. Clip negative expression and renormalize implied libraries. Both new support and normalization can alter gene covariance, which the held-out distribution/variogram checks must assess.

The [pinned public core scorer](https://github.com/aristoteleo/veckit) supplies metric definitions. Local panels use all atlas genes and their own copy-last floor and split-cell ceiling, with the four T1 weights. Reject any calibration whose ceiling is not better than its floor; do not drop an unfavorable metric to obtain a passing score. No official anchors or returned leaderboard scores determine forecast content. Neither a local score nor the local >72 gate verifies the hidden E10.5 leaderboard score. Atlas sampling/dissection and the 27,669-gene panel differ from the challenge data and official 32,285-gene panel.

## Artifacts and checks

- Acquisition: `private/extended_atlas/manifest.json`.
- Prepared data: `private/extended_prepared_02/`, including fixed donor indices, genes, normalization/source checksums and events.
- Training: `private/extended_training_01/`, including the frozen plan, actual execution events, NumPy model checkpoints, selected configuration and metric report when complete.

```powershell
outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/acquire_extended_atlas.py --download
outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/prepare_extended_atlas.py
outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/train_extended_atlas.py
outputs/research_workflow/.venv/Scripts/python.exe -m unittest discover -s outputs/t1_iterations -p test_extended_atlas.py
```

Preparation/training directories cannot be reused silently. The tests check binary decoding and invalid records, future-stage invariance of fitted representation/decoder/pseudo-pair velocities, and implied-library normalization. These checks establish engineering properties, not future-stage accuracy. Requirements are pinned in `requirements-local-lock.txt`.
