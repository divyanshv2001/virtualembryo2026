# Distribution audit and rolling development

The rejected E9.5 latent rollout kept implied libraries at 10,000 but reduced the zero fraction from 86.82% in its E8.5 donors to 48.02%. It activated 38.89% of all entries that were zero in the corresponding donor and produced negative pre-clipping values in 20.12% of entries. Covariance changed by 55.16% in Frobenius norm over a fixed 128-feature diagnostic subset. These are descriptive diagnostics; they do not prove that support changes alone caused the score failure. Comparisons of individual target rows with donor rows do not represent lineage pairs.

The first 256 cells in each challenge input have implied-library totals approximately 10,000 and zero fractions of 87.76% (E8.5) and 87.54% (E9.5). This sample confirms compatibility with the atlas normalization convention, not matched biological populations. The whole-embryo E9.5 atlas sample includes just 16 explicitly labelled cardiomyocytes, versus a cardiac-enriched challenge dataset. Atlas sampling and dissection remain major transfer limitations.

## Constraints and validation

`constrained_forecast.py` fits training-only PCA on 384 variance-selected genes. Recent stage means estimate a latent population drift; regularized covariance converts that drift into capped donor weights. Systematic resampling retains measured expression vectors and their covariance structure within donors. Optional bounded multiplicative abundance changes preserve donor zero patterns, followed by full-panel library normalization. These constraints are deliberate hypotheses, not claims that future biological expression cannot activate genes.

`state_population.py` replaces global trends with 24 past-only k-means states. Population proportions are smoothed before log-frequency extrapolation. State-specific abundance slopes require at least eight cells at each of the three recent training stages and shrink with sample size. States without sufficient support retain donor expression.

Both whole-atlas batches froze seven configurations before their evaluations. Five rolling development splits were used: E7.5→E7.75, E7.5→E8.5, E8.0→E8.25, E8.0→E9.0 and E8.5→E9.5. Every fitted feature selection, scaler, PCA, state and trend excludes that fold's future stages. All these panels are development checks; previously evaluated E8.5/E9.5 are not fresh blind tests. Random-cell halves provide sampling ceilings, not independent-embryo replication. All four calibration checks must pass.

| Whole-atlas configuration | E7.75 | E8.5 | E8.25 | E9.0 | E9.5 | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 |
| Global population tilt 0.25 | 53.74 | 50.79 | 52.17 | 50.07 | 56.94 | 52.74 |
| Global population tilt 0.5 | 55.38 | 50.83 | 52.06 | 49.77 | 56.98 | 53.01 |
| Global expression trend 0.1 | 51.72 | 51.35 | 51.44 | 49.53 | 57.10 | 52.23 |
| State expression trend 1.0 | 51.29 | 52.83 | 49.46 | 49.44 | 55.43 | 51.69 |

The distribution constraints improved the previously failed E9.5 proxy (34.98), but the new candidates are far below 72 and several fail to beat persistence on E9.0. None passed the declared promotion gate: valid calibration and aggregate >50 on every fold, each metric skill ≥50 on every fold, and mean >72. No challenge candidate was exported or submitted. No Jev call was made.

## Mapping and transfer

Only 26,775 official genes have unambiguous exact atlas-symbol matches. The atlas lacks 5,489 official genes and has 26 duplicated nonempty symbols; 21 official genes are ambiguous under this mapping. The prospective transfer policy retains challenge donor expression for missing or ambiguous genes and transfers only unique exact matches. An actual transfer implementation must specify the effect of any subsequent normalization on retained genes and validate the complete 32,285-gene file. This has not been bypassed by zero filling.

## Reproduce

```powershell
outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/audit_rollout.py
outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/rolling_constrained.py --round NEW_ROUND
outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/rolling_constrained.py --round NEW_STATE_ROUND --family state
outputs/research_workflow/.venv/Scripts/python.exe -m unittest discover -s outputs/t1_iterations -p test_constrained_forecast.py
```

Existing private rounds cannot be overwritten. Plans, actual execution events, executed code, donor indices, checkpoints and full metric reports remain in ignored `private/rolling_constrained_01`, `private/rolling_state_01` and `private/rollout_audit_01`. Three forecast tests and the three existing atlas tests passed, checking future-data invariance, unchanged persistence, bounded resampling weights, zero-support preservation, binary decoding and library totals.

## Completed cardiac-cohort correction

Prepared `private/cardiac_prepared_01` from the already downloaded, checksum-verified source. Ten fixed published cardiac/progenitor/endocardial/epicardial annotations define the cohort. Include E7.5–E9.5 only; mixed stages remain excluded. Seeded uniform sampling retains up to 1,000 cells per stage, keeping every cell in smaller pools. The result contains 7,399 cells × 27,669 genes: 68 at E7.5, 435 at E7.75, 896 at E8.0, and 1,000 at each subsequent quarter-day stage. Expression SHA-256: `a8c2ff0e1675e58f45c96aca868a17ebba634b6ba5f9c98f8dea4ae210d6ca34`.

Annotations select the cohort and are not learner features. Published annotations may have used a jointly analyzed atlas, so this does not establish independent prospective dissection accuracy. The cardiac subset omits several cell populations present in the challenge and is not an exact substitute for the challenge distribution. Its ceilings and floors differ from the whole-embryo panels; cross-cohort score differences are not direct comparisons of accuracy.

Two more seven-configuration batches completed four splits each: E8.0→E8.25, E8.0→E9.0, E8.25→E8.5 and E8.5→E9.5. The sparse earliest cardiac stage prevents a well-supported three-stage fit ending at E7.5. No future expression enters any fitted representation or trend. All calibration checks passed, but no candidate passed the promotion gate.

| Cardiac configuration | E8.25 | E9.0 | E8.5 | E9.5 | Mean |
|---|---:|---:|---:|---:|---:|
| Persistence | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 |
| Global population tilt 0.25 | 52.37 | 50.23 | 50.28 | 60.51 | 53.35 |
| Global population tilt 0.5 | 53.48 | 50.12 | 50.25 | 61.20 | 53.76 |
| Global expression trend 0.25 | 51.36 | 49.66 | 48.16 | 67.32 | 54.13 |
| Global tilt 0.5 + expression 0.1 | 53.16 | 50.18 | 49.93 | **68.07** | **55.34** |

The best mean candidate reaches 68.07 on the E9.5 cardiac development panel but dips below persistence on E8.5. State-conditioned candidates also fail promotion; their best E9.5 score is 59.25. Tiny earlier-state pools limit state-specific trends, and guarded unsupported states retain expression. The result supports keeping distribution constraints and investigating more robust cohort/state coverage, not promoting a single favorable stage or increasing strength until one panel passes.

Across all four batches, **126 model/fold evaluations** completed, including persistence controls. No model reached the declared 72 gate, no challenge prediction was exported, and zero official submissions or Jev requests were used. Later branches must retain these results as development history. Transfer and full-file validation remain conditional on a candidate passing the local checks.

```powershell
outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/prepare_extended_atlas.py --round NEW_CARDIAC_DATA --domain cardiac
outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/rolling_constrained.py --round NEW_CARDIAC_GLOBAL --dataset NEW_CARDIAC_DATA
outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/rolling_constrained.py --round NEW_CARDIAC_STATE --dataset NEW_CARDIAC_DATA --family state
```

The completed cardiac rounds are `private/cardiac_global_01` and `private/cardiac_state_01`, containing their frozen plans, actual execution logs, fitted checkpoints, donor indices and full metric reports.

These local panels are not the official hidden E10.5 evaluation and cannot certify a leaderboard score of 72. The four remaining user-reported daily submission slots are conserved.
