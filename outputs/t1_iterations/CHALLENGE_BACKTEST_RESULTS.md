# Full-panel challenge-data backtest

Completed the actual challenge-population development test: learn atlas trends through E8.5, anchor on challenge E8.5 cells, and predict the supplied challenge E9.5 cells. The best configuration averaged **51.08** against persistence at 50. The 72 objective remains unmet; no model was refitted for E10.5, no E10.5 submission was exported, and no submission or Jev request was used.

## Fitting and transfer

Both runs train on 13,963 cells from `associated_prepared_01` at stages through E8.5. Later atlas rows never enter feature variances, PCA, state fitting, trends or alignment. The fixed challenge anchor comprises 1,500 E8.5 donors, sampled with seed 20260928. Challenge E9.5 expression is first opened for evaluation after every candidate prediction is saved and hashed. Its file checksum is recorded earlier solely for input integrity, not fitting.

The model fits a training-only 384-feature PCA and four/eight k-means states. Feature selection is restricted to unique exact symbols shared with the official challenge panel, so no missing atlas feature is imputed into the representation. All outputs retain the official **32,285-gene order**. Transfer changes only 26,775 uniquely mapped genes; 5,510 missing or ambiguous official genes retain their corresponding donor values exactly. Rescaling occurs only within mapped abundance mass, preventing normalization from altering protected genes.

Two declared controls test transfer adaptation. `challenge_backtest_01` aligns past atlas/challenge feature means and standard deviations, with scale ratios clipped to [0.5,2]. `challenge_identity_01`, added after the first run's weak results, projects measured challenge features directly into the atlas representation. It uses identical donors, target panels, configurations and scorer seeds. Neither adaptation uses E9.5 expression. Both apply a training-source distance gate; state forecasts also retain the effective-sample-size and covariance guards.

The identity control trusts 95.67% of anchor donors for its best configuration. Untrusted cells retain unchanged expression factors. This is a source-support diagnostic, not proof of correct state assignment or ancestry.

## Evaluation and outcome

Each run freezes nine configurations, then evaluates three predeclared sampling panels (seeds 20260928, 20260929 and 20260930). Each panel draws 2,000 E9.5 cells and splits them into 1,000 truth and 1,000 ceiling cells. The 1,500 fixed E8.5 donors provide the reference and persistence floor. All four metrics use the complete official gene panel and their usual weights. All 54 candidate/panel calibrations passed the finite/better-ceiling checks.

These are sampling panels of one observed temporal transition. Panels can overlap and are not independent embryo replicates. E9.5 has been inspected in earlier development, so this is not a fresh blind test. The local E9.5 anchors differ from the official hidden E10.5 anchors; scores cannot certify or estimate an official leaderboard score of 72.

| Configuration | Mean/std alignment: mean local score | Direct projection: mean local score |
|---|---:|---:|
| Persistence | 50.00 | 50.00 |
| Incumbent global population/expression | 48.10 | 48.53 |
| Global population only | 49.22 | 49.89 |
| Global expression 0.1 | 49.35 | 49.39 |
| Global expression 0.25 | 47.49 | 47.58 |
| Four-state population only | 48.51 | 49.77 |
| Four-state expression 0.25 | 48.98 | 49.73 |
| Eight-state population only | 49.62 | 50.00 |
| Eight-state expression 0.25 | **50.72** | **51.08** |

The best direct-projection candidate scored **51.16, 51.40 and 50.66** across the three panels. Its mean component skills were DE recovery 51.68, direction 51.46, MMD 50.77 and variogram 50.30. This is a small observed gain, not evidence of a large or statistically independent improvement. Dropping alignment helped slightly but did not resolve the weak transfer. The incumbent and global expression trends remain below persistence.

No candidate passes the declared development gate: every panel >72 and every component ≥50. Earlier temporal promotion evidence also remains absent. Consequently no future refit/export is performed. Preserve these negative results rather than selecting a favorable atlas-local panel or presenting the prior 68.07 cardiac proxy as a challenge score.

## Files and verification

Each private run retains its frozen plan, actual execution events, executed code, two model checkpoints, donor indices, full-panel predictions, evaluation-row selections and complete metric report. The best observed-stage prediction is saved as `BACKTEST_ONLY_E9.5__state_k8_e0.25.h5ad`, explicitly annotated as a development backtest. **It is not an E10.5 submission.** Both backtest files pass the local format validator: 1,500 × 32,285, float32, exact panel, finite nonnegative values and no coordinates or source observation annotations.

An additional check compared all 18 generated candidates against their actual resampled donors. Every protected gene was bit-identical, every zero pattern was preserved, and maximum implied-library changes were 0.000451 (alignment) and 0.000298 (direct projection) on a total near 10,000. Both input-file checksums were unchanged at completion. No raw datasets, predictions or model checkpoints are committed.

Nine relevant tests passed: two transfer-forecast tests, three prior population tests, two robust-population tests and two gene-mapping tests. They check future-stage invariance, direct projection semantics, eligible-feature selection, protected genes, persistence, zero support, covariance bounds and library conservation. They do not verify the official score.

```powershell
outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/challenge_backtest.py --round NEW_ALIGNED_RUN
outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/challenge_backtest.py --round NEW_DIRECT_RUN --alignment identity
outputs/research_workflow/.venv/Scripts/python.exe -m unittest discover -s outputs/t1_iterations -p test_challenge_transfer.py
```

Existing runs cannot be overwritten. Completed records are under ignored `private/challenge_backtest_01` and `private/challenge_identity_01`. Further research can use these as development evidence, while hidden E10.5 remains unavailable. The four remaining user-reported submission slots remain unused by this work.
