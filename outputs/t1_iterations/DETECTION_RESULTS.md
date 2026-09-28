# Detection and local-maturation development experiments

Objective remains unfinished: no candidate crossed 72. All results below are local E9.5 development scores, not official E10.5 leaderboard scores. No submissions or Jev calls were used.

Every batch froze configurations and three evaluation seeds before fitting/forecasting. Training used atlas stages <=E8.5 and challenge E8.5 anchors only; every forecast was saved before reading E9.5 expression. Evaluation used the complete 32,285-gene panel and unchanged four-metric calibration. E9.5 is reused development data; overlapping cell panels are not independent embryos.

| Batch / candidate | Mean | Minimum |
| --- | ---: | ---: |
| detection_backtest_01 / copy_last | 50.0000 | 50.0000 |
| detection_backtest_01 / zero_preserving_k8 | 51.0759 | 50.6642 |
| detection_backtest_01 / detection_0.25_cap0.02 | 51.1958 | 50.8157 |
| detection_backtest_01 / detection_0.5_cap0.02 | 51.2266 | 50.8520 |
| detection_backtest_01 / detection_1.0_cap0.05 | 51.0759 | 50.6642 |
| hurdle_backtest_01 / copy_last | 50.0000 | 50.0000 |
| hurdle_backtest_01 / detection_incumbent | 51.2266 | 50.8520 |
| hurdle_backtest_01 / positive_abundance_0.25 | 51.0689 | 50.8446 |
| hurdle_backtest_01 / positive_abundance_0.5 | 51.0689 | 50.8446 |
| hurdle_backtest_01 / positive_abundance_1.0 | 51.0689 | 50.8446 |
| guard_backtest_01 / copy_last | 50.0000 | 50.0000 |
| guard_backtest_01 / detection_incumbent | 51.2266 | 50.8520 |
| guard_backtest_01 / positive_e0.5_cov0.2 | 51.1309 | 50.8384 |
| guard_backtest_01 / positive_e1.0_cov0.4 | 51.0685 | 50.7222 |
| guard_backtest_01 / joint_e0.5_cov0.2 | 51.3538 | 50.9453 |
| guard_backtest_01 / joint_e1.0_cov0.4 | 51.4328 | 51.0390 |
| neighborhood_backtest_01 / copy_last | 50.0000 | 50.0000 |
| neighborhood_backtest_01 / zero_preserving_k8 | 51.0759 | 50.6642 |
| neighborhood_backtest_01 / neighborhood_s0.25_a0 | 47.8866 | 47.7781 |
| neighborhood_backtest_01 / neighborhood_s0.25_a32 | 47.4397 | 47.3039 |
| neighborhood_backtest_01 / neighborhood_s0.5_a64 | 43.8110 | 43.5830 |
| neighborhood_backtest_01 / neighborhood_s1.0_a256 | 39.9441 | 39.5528 |

Detection changes at strength 0.5/cap 0.02 activated 8,372 and removed 4,767 donor entries, improving the three-panel mean from 51.0759 to 51.2266. The largest detection setting failed the 0.1 covariance guard and reverted to the control. All 5,510 unmapped/ambiguous genes remained bit-for-bit unchanged; audited maximum library difference was <0.0003 on approximately 10,000-count normalized libraries.

Separating positive-cell abundance from detection did not improve results. The 0.1 covariance guard collapsed tested positive-abundance strengths to the same effective value; declared 0.2/0.4 guard ablations confirmed that stronger conditional-positive abundance still did not beat the joint-abundance model. The best joint setting reached 51.4328, minimum 51.0390. The scorer and its calibration were unchanged.

Local nearest-neighbor maturation used 32 past-only neighbors per cell at each of three stages, suppressed sign reversals, capped additive changes and limited new positives per cell. It worsened scores even without activations; its strongest setting fell below 40 on two panels. This branch is rejected rather than promoted.

All recorded calibrations were valid. Backtest artifacts passed shape/panel/finite-value validation and are explicitly labeled observed E9.5 only. No future refit/export occurred.

The frozen 16-replicate Monte Carlo pilot completed with all calibrations valid. Joint setting: mean 51.2530, empirical 2.5th percentile 50.9660; detection incumbent: mean 51.0715, lower tail 50.7086. All four mean skills exceeded 50, but remained below 52.5. These are conditional cell-sampling stability summaries, not biological confidence intervals, and cannot satisfy the >=64-replicate final gate. The five-stage population-window ablation completed. All three tilt levels produced identical predictions after clipping/backoff and averaged 50.1380, below the unchanged joint incumbent (51.4328). Longer-window composition changes did not help on these panels. See LOCAL_OPTIMIZATION_STATE.json for the current execution checkpoint.

Validation: 10 focused tests passed across source exclusion, protection of unmapped genes, mass conservation, sparse activation bounds, incumbent restoration and Monte Carlo gate checks.

The next search tests 4, 8, 16 and 32 past-only states under standard or unit feature scaling. Its eight model configurations, three evaluation seeds and unchanged metric calibration are frozen in private/state_search_01/plan.json. Forecasts are generated before target expression is opened. Generation and scoring checkpoint separately; --resume validates source/input hashes and skips completed forecasts/panels. Biological arrays stay private. This is another development search, not proof of a >72 score.
