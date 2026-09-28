# Initial synthesis for hostile review

All 16 independent discipline reviews and 28 initial proposals are saved. This synthesis draws on their actual reviews, not inherited persona history. All eight two-sided cross-examinations are recorded in disagreement_registry.json; issues are not forced into consensus. Exactly two critic/revision loops follow this version.

## Scientific problem and diagnosis

The intended problem is distributional forecasting of measured embryonic expression, joint expression/geometry, and perturbation response. The supplied project implements orchestration, not a biological predictor. Source inspection, zero-data inventory, 15-passing/one-failing test suite, and four synthetic defect reproductions support that diagnosis. Hardcoded scores cannot substitute for artifacts. No performance, state-of-the-art result, identified cell lineage or causal mechanism is demonstrated.

The prior literature already supplies temporal OT, spatial mapping, conditional generative models, dynamical predictors and graph-based perturbation methods. Adding these ingredients alone is not a demonstrated scientific gap or novelty. A more defensible question is what predictive information remains after separating population composition, conditional expression state and spatial organization under independently held-out samples. Useful error attribution and a valid benchmark could be a contribution, but novelty requires a dedicated closest-work comparison and an empirical result.

## Proposed hypothesis

Under a verified observation contract, a small conditional-state model may improve held-out measured-expression forecasts beyond persistence and composition-only resampling. For T2, jointly preserving or predicting expression-position dependence may add value beyond separately matched marginals. These are hypotheses, not expected leaderboard wins. T3 gene-specific generalization is an independent unproven extension: one Mab21l2 identity does not identify Gata4 or β-catenin effects.

## Formulation and smallest useful model

For task/assay a, age t, genotype g and sampled tissue h, define a predictive population distribution

`p(X,C | t,g,a,h) = Σ_k π_k(t,g,a,h) p(X | k,t,g,a,h) p(C | X,k,t,g,a,h)`.

Omit C for T1. k is a training-derived broad state, with soft membership and unknown/out-of-support diagnostics. This factorization is a statistical convention, not a causal DAG or identified developmental decomposition. With measured means μ_kg and weights π_k, aggregate changes contain composition, conditional-state and interaction terms. Observation selection/capture can change all of them.

Start with whole-row empirical resampling. Add training-only temporal weights, then shrunk affine or low-rank state shifts while preserving residual covariance. Preserve nonnegative normalized output and audit clipping. For T2 separate calibrated physical scale from rigid frame nuisance and optional conditional deformation. Do not whiten away physical growth or force nonlinear correspondence across changing anatomy. An expressive decoder does not validate a future population absent from training support.

Do not initially fit a genotype-response coefficient for an unseen gene from one KO and call it identified. Compare identity and generic/eligible-prior baselines for descriptive predictive experiments only. Any prior needs measured-source and pretrained exposure screening. Protected target response measurements remain quarantined from fitting/tuning/selection.

Escalate to VAE/OT/flow/ODE/GNN only if simpler models leave a relevant residual. Match model capacity, tuning access, sample counts and compute accounting. Missing biological data currently prevent all fitting and tests.

## Evaluation and evidence standard

EXPERIMENT_DESIGN.md specifies composition-only/state-only/combined/persistence and spatial extensions. T2's three released snapshots can give a limited interpolation or pseudo-future test. T1's two snapshots cannot validate a learned two-point trend after a stage holdout. T3's one released intervention cannot validate arbitrary unseen-gene transfer. A local proxy is not evidence of real-board improvement; no proxy score is subtracted from a leaderboard baseline.

The primary contrast is combined versus composition-only raw MMD, with all official metric components reported. Avoid degradation in direction, covariance, shape and neighborhoods. Independent embryos—not cells or sampling seeds—support biological uncertainty. Pilot variance supports power planning only when replicated and matched to the scientific alternative. Training transforms remain fold-contained; target-fitted scorer PCA/probe is evaluation-only. Geometry and topology need orthogonal measurements where metrics have blind spots. Replication, allele/timing verification, full-transcriptome readouts, imaging, lineage tracing and rescue support progressively stronger claims.

Cross-examination has narrowed the proposals: separate acquisition-order bias from subset-neighborhood reconstruction; require covariance-preserving conditional-state baselines before flow comparison; preserve capture/abundance nonidentifiability and abstention; treat anatomy, scorer invariance and topology as separate endpoints; require representation/decoder controls and orthogonal state readouts. Logical counterexamples in logical_check_results.json were executed on arbitrary values, not biological data. Protected-target measured literature encountered during this research audit is exposure-unresolved and quarantined from inference; relabeling a packet cannot erase agent exposure. Current run is not certified competition-eligible.

## Literature basis and uncertainty

Schiebinger et al. (Cell 2019) and moscot (Nature 2025) justify testing assumption-dependent transport while rejecting uniquely inferred ancestry. TrajectoryNet/PRESCIENT and 2025/2026 dynamics papers illustrate explicit model priors, not validated assumptions here. scGen/GEARS motivate response transfer, but their training coverage differs substantially. Ahlmann-Eltze et al. (Nature Methods 2025), Systema (online 2025 / volume 2026), and Wei et al. (Nature Methods 2026) motivate simple baselines and perturbation-specific/context-specific evaluation. Source verification levels are recorded in expert reviews and LITERATURE_NOTES.md; inaccessible full text is not silently treated as read.

## Current weaknesses the critic must attack

Unknown replicate structure/units/scorer implementation; sparse stage and intervention coverage; composition/annotation/capture ambiguity; unsupported future states; architecture selection on target residuals; unspecified practical improvement and secondary margins; uncertainty beyond observed support; hidden target exposure through priors; no novelty evidence. Code defects include invalid authority on empty API answers, duplicate pruning order, unreviewed action queues and critic-to-empirical lifecycle conflation. Preserve unresolved risks and do not invent resolution through expert agreement.
