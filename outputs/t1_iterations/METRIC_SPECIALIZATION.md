# Metric specialists and integration

The strategy is feasible as a constrained multi-objective search: separate specialist predictions can be complementary, but their best scores do not add. Integration must be rescored on all four metrics. These are reused E9.5 development results, not official leaderboard performance or independent biological validation.

Fifty-four candidates from eight completed full-panel batches were compared only after verifying identical reference/truth calibration panels. Two candidates form the non-dominated front.

| Candidate | DE recovery | Direction | MMD | Variogram | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard_k16 | 52.58 | 55.95 | 52.42 | 51.24 | 53.11 |
| unit_k16 | 52.21 | 58.10 | 57.00 | 53.16 | 55.31 |

The standardized 16-state model leads in DE recovery. The unit-scaled 16-state model leads in the other three metrics and the aggregate. This validates maintaining component-specific evidence; the small DE difference alone does not justify discarding the stronger joint model.

The first integration test mixed complete cells from these two forecasts at four declared DE-specialist weights. Whole-cell rows and provenance were retained; no RNA values or metric scores were averaged. Mixtures and both constituents were frozen before reading target expression in this run. Selection used previous E9.5 development scores, so this is not a blind validation.

| Integration candidate | Mean | Minimum |
| --- | ---: | ---: |
| copy_last | 50.0000 | 50.0000 |
| de_specialist | 53.1072 | 52.3410 |
| joint_specialist | 55.3123 | 54.1990 |
| mixture_de_0.1 | 54.6825 | 53.5782 |
| mixture_de_0.25 | 54.4332 | 53.4449 |
| mixture_de_0.5 | 54.1697 | 53.1637 |
| mixture_de_0.75 | 54.8128 | 53.8852 |

None of these mixtures beat the joint constituent (55.3123). Reject these combinations; do not report a synthetic score assembled from specialist maxima. This negative result does not disprove all shared-model or ensemble designs. It shows these specific predictors/mixtures are insufficient.

Next executable batch: private/de_specialist_search_01, testing bounded abundance factors 1.5/2/4 against the unchanged 1.25 control under two past-only 16-state representations. Primary objective is DE recovery, with complete vectors retained and non-primary mean skills >=50 required for joint eligibility. The scorer requires absolute mean-log shifts >=0.25, so the existing 25% abundance cap can constrain many genes from qualifying; normalization and cell composition also affect those shifts. Scoring anchors and thresholds are unchanged.

The final >72 Monte Carlo/temporal gate, full gene panel, no-future-fitting rule and submission conservation remain unchanged. No official uploads or Jev requests occurred.

Updated component results after integration and factor-cap testing: the 75% DE-specialist cell mixture increased mean DE recovery to 53.1215, above either initial constituent (52.5824 and 52.2131), but its joint score was 54.8128, below 55.3123. The factor-cap-2 candidate increased MMD slightly to 57.1390 while lowering other components, so its overall mean was 55.0513. These measured trade-offs support retaining separate specialist/Pareto evidence while requiring integrated gains; they do not establish a >72 result.

Larger abundance caps did not improve the declared DE objective or joint incumbent. That hypothesis is rejected as a solution for this batch. The next active job is a frozen 16-replicate stability pilot comparing the stronger unit_k16 model with standard_k16. It cannot satisfy the >=64-replicate final gate; further mechanism changes and temporal checks remain necessary.
