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

Existing private rounds cannot be overwritten. Plans, actual execution events, executed code, donor indices, checkpoints and full metric reports remain in ignored `private/rolling_constrained_01`, `private/rolling_state_01` and `private/rollout_audit_01`. Three tests passed, checking future-data invariance, unchanged persistence, bounded resampling weights, zero-support preservation and library totals. A cardiac-cohort branch is being prepared from the same already downloaded source; its labels restrict the cohort rather than become learner features.

These local panels are not the official hidden E10.5 evaluation and cannot certify a leaderboard score of 72. The four remaining user-reported daily submission slots are conserved.
