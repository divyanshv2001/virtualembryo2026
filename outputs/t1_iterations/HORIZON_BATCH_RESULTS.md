# One-day source-cohort audits

Two observed one-day folds, E8.0→E9.0 and E8.25→E9.25, use the same cardiac-associated source cohort, 1,500 donors and complete official gene ordering. Each forecast is frozen before later-stage expression is read for scoring. Missing/ambiguous genes use zero placeholders in this atlas-only diagnostic; these are not measured challenge genes. These folds therefore improve horizon matching but do not certify challenge-domain or embryo-independent generalization.

The prior model scores 47.143 and 47.987. All component ablations also score below persistence on both folds. This confirms one-day failure within the source cohort; it cannot isolate hidden E10.5 causes.

The positive-quantile mechanism fits three historical positive-expression quantile distributions, suppresses reversing trends, shrinks low-support slopes, and preserves the donor zero mask and number of cells. Per-cell mapped mass and protected genes are guarded. Marginal proposals are monotone; cell-specific mass conservation can change final cross-cell ranks.

A second pilot conditions those margins on four broad past-fitted states. It retains fixed donor counts, changes only trusted anchors, requires 20 anchors and 20 source cells at every recent stage, and applies both within-state and overall covariance guards. It tests composition confounding, not inferred cell proliferation.

Literature: [Schefzik, Thorarinsdottir and Gneiting (2013)](https://arxiv.org/html/1302.7149v2), methods 4.1–4.3 and experiments 5.4 read. ECC separates marginal calibration from rank dependence. Its weather experiments found benefits dependent on the dependence structure. This implementation adapts that separation to historical positive scRNA margins; it is neither faithful ECC nor evidence that ECC improves this challenge.

| Run | Candidate | Fold 1 | Fold 2 | Mean |
| --- | --- | ---: | ---: | ---: |
| matched_horizon_audit_01 | copy | 50.000 | 50.000 | 50.000 |
| matched_horizon_audit_01 | unit16 | 47.143 | 47.987 | 47.565 |
| matched_horizon_audit_01 | expression_only | 49.122 | 47.039 | 48.081 |
| matched_horizon_audit_01 | population_only | 47.256 | 48.903 | 48.079 |
| matched_horizon_audit_01 | population_detection | 47.126 | 48.153 | 47.640 |
| quantile_horizon_pilot_01 | copy | 50.000 | 50.000 | 50.000 |
| quantile_horizon_pilot_01 | unit16 | 47.143 | 47.987 | 47.565 |
| quantile_horizon_pilot_01 | quantile_0.25 | 50.975 | 49.091 | 50.033 |
| quantile_horizon_pilot_01 | quantile_0.5 | 50.854 | 48.980 | 49.917 |
| quantile_horizon_pilot_01 | quantile_1.0 | 50.861 | 48.762 | 49.811 |
| state_quantile_horizon_pilot_01 | copy | 50.000 | 50.000 | 50.000 |
| state_quantile_horizon_pilot_01 | unit16 | 47.143 | 47.987 | 47.565 |
| state_quantile_horizon_pilot_01 | quantile_0.25 | 50.592 | 48.203 | 49.398 |
| state_quantile_horizon_pilot_01 | quantile_0.5 | 50.592 | 48.072 | 49.332 |
| state_quantile_horizon_pilot_01 | quantile_1.0 | 50.596 | 47.987 | 49.292 |

Status: completed. Full four-metric vectors, raw metrics, model audits and provenance hashes are in HORIZON_BATCH_RESULTS.json.
No new prospective export or official submission. The >72 objective remains unfinished.
