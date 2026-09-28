# One-day source-cohort audits

Two observed one-day folds, E8.0→E9.0 and E8.25→E9.25, use the same cardiac-associated source cohort, 1,500 donors and complete official gene ordering. Each forecast is frozen before later-stage expression is read for scoring. Missing/ambiguous genes use zero placeholders in this atlas-only diagnostic; these are not measured challenge genes. These folds therefore improve horizon matching but do not certify challenge-domain or embryo-independent generalization.

The prior model scores 47.143 and 47.987. All component ablations also score below persistence on both folds. This confirms one-day failure within the source cohort; it cannot isolate hidden E10.5 causes.

The positive-quantile mechanism fits three historical positive-expression quantile distributions, suppresses reversing trends, shrinks low-support slopes, and preserves the donor zero mask and number of cells. Per-cell mapped mass and protected genes are guarded. Marginal proposals are monotone; cell-specific mass conservation can change final cross-cell ranks.

A second pilot conditions those margins on four broad past-fitted states. It retains fixed donor counts, changes only trusted anchors, requires 20 anchors and 20 source cells at every recent stage, and applies both within-state and overall covariance guards. It tests composition confounding, not inferred cell proliferation.

The annotation pilots group cells by observed published type strings and estimate positive-abundance and detection trends separately. Four ablations compare expression, detection and their combination. A second frozen pilot raises the source support requirement from 20 to 100 cells per type per past stage and the positive-expression minimum from10 to20. Donor counts stay fixed. Annotations now enter learner grouping; joint atlas annotation is a limitation, and later-stage label values are excluded from fitting.

Literature: [Schefzik, Thorarinsdottir and Gneiting (2013)](https://arxiv.org/html/1302.7149v2), methods 4.1–4.3 and experiments 5.4 read. ECC separates marginal calibration from rank dependence. Its weather experiments found benefits dependent on the dependence structure. This implementation adapts that separation to historical positive scRNA margins; it is neither faithful ECC nor evidence that ECC improves this challenge.

The [muscat paper](https://www.nature.com/articles/s41467-020-19894-4) motivates separating within-subpopulation state changes from differential abundance. Introduction, simulation results (20–400 cells) and simulation-preprocessing methods were read. It found sizable detection gains between20 and100 cells per subpopulation/sample. It uses replicated samples and count-based inference; our normalized-abundance trend forecast does not reproduce those methods, biological replication or their inferential guarantees.

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
| annotation_horizon_pilot_01 | copy | 50.000 | 50.000 | 50.000 |
| annotation_horizon_pilot_01 | unit16 | 47.143 | 47.987 | 47.565 |
| annotation_horizon_pilot_01 | annotation_expression | 49.595 | 47.855 | 48.725 |
| annotation_horizon_pilot_01 | annotation_detection | 48.525 | 45.679 | 47.102 |
| annotation_horizon_pilot_01 | annotation_combined | 49.336 | 46.920 | 48.128 |
| annotation_horizon_pilot_01 | annotation_detection_full | 49.311 | 47.038 | 48.175 |
| annotation_support_pilot_01 | copy | 50.000 | 50.000 | 50.000 |
| annotation_support_pilot_01 | unit16 | 47.143 | 47.987 | 47.565 |
| annotation_support_pilot_01 | annotation_expression | 49.325 | 47.822 | 48.574 |
| annotation_support_pilot_01 | annotation_detection | 48.895 | 45.923 | 47.409 |
| annotation_support_pilot_01 | annotation_combined | 49.345 | 46.998 | 48.172 |
| annotation_support_pilot_01 | annotation_detection_full | 49.448 | 47.122 | 48.285 |

Status: completed. Full four-metric vectors, raw metrics, model audits and provenance hashes are in HORIZON_BATCH_RESULTS.json.
No new prospective export or official submission. The >72 objective remains unfinished.
