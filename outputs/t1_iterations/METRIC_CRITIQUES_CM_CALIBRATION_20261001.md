# CM calibration: specialist critiques and next decision

Batch: cm_capture_calibration_fullpanel_01. All 12 full-panel scores/calibrations retained. No candidate passed promotion. These are methodological specialist agents, not a claim of human credentials. User requested the role update after the initial reviews: each specialist now provides a problem, concrete proposed solution and validation. Revisiting these critiques does not create new experiments or rewards. Role policy version 2.

Copy scores: 50/50. Refitted anchor: 51.063292/50.401927. CM .25: 51.068611/50.437171. Its first-fold DE skill decreases by .000308; its second-fold DE skill remains below copy (.446429 versus .5). Tiny headline gains do not satisfy the four-metric or original readiness gates. Separate reward moves -436 to -475, not a benchmark score.

## DE recovery specialist

Problem: CM corrections barely change first-fold recovery and remain worse than persistence in the second. Incorrect effect signs or ordering are plausible, but aggregate metrics cannot determine the cause.

Proposed solution: a past-only gene-wise effect-reliability shrinkage gate, learned from earlier eligible captures, with current capture excluded from base learning. Shrink unreliable changes toward persistence; current anchors calibrate only. This is an untested proposal. Prior empirical-Bayes/reliability ablations already exist and must be checked before declaring a distinct implementation.

Validation: gated CM, unchanged CM, anchor and copy on identical genes/folds. Require recovery to exceed copy and avoid anchor regression in both folds. No future-target ranks or score inversion may fit the gate.

## DE direction specialist

Problem: global offsets regress on the first fold; CM .25 direction improves both by negligible amounts. Broad detection adjustments may perturb useful expression-change signs, but mechanism/significance are unestablished.

Proposed solution: apply fixed CM .25 offsets only where past capture-balanced detection and mean-expression trends agree with the implied expression change. Freeze support and uncertain/zero-trend handling in advance. This is untested; earlier sign-gated and capture-balanced failures constrain interpretation.

Validation: gated versus ungated CM .25 and anchor, identical refits/seeds. Require positive paired direction changes on both folds with capture-level uncertainty; regression or inadequate support blocks promotion. Few captures cannot establish independent-embryo confidence.

## MMD specialist

Problem: global offsets trade first-fold regression for second-fold gain; CM offsets barely affect whole-population distance. Hypothesis: unnecessary non-CM sparsity changes and small CM population weight limit distribution gains.

Proposed solution: lineage-specific detection-logit residuals adjusted for observed library size, shrunk using permitted past capture support. Fit shrinkage only on past-held-capture splits, preserve positive heads and lineage proportions. Untested; prior CDR/coupling work must be checked for duplication.

Validation: matched copy/anchor/CM .25, identical cells/seeds/kernels. Whole-population and CM-only MMD are separate headline/diagnostic measurements. Require whole-population gain on both temporal folds without non-CM regression. No independent-embryo claim.

## CSS variogram specialist

Problem: stronger/global offsets can worsen nonspatial cell-state contrasts; CM .25 gains are tiny. Aggregate CSS does not identify biological versus capture effects.

Proposed solution: penalize CM odds corrections that distort within-lineage pairwise cell-state contrasts on permitted past/current observations, smooth sparse-gene odds and choose the penalty on past-only capture exclusions. Untested; scoring/calibration must remain unchanged.

Validation: matched copy/anchor/unregularized CM .25, same cells/features/calibration. Require lower full-panel raw CSS on both folds, improvement over CM .25 and a predeclared contrast tolerance. Any fold regression or future-dependent tuning rejects the candidate.

## Selected immediate action

Jev selected fixed_capture_replication, confidence .94, one 1859-byte advisory with actual 886 input/54 output tokens. Freeze CM .25 before testing second-largest current captures: sample33 at E8.0 (699 cells/44 CM), sample24 at E8.25 (1108 cells/134 CM). Refit all base components excluding that capture; permit its current anchors, compare copy/anchor/CM .25 on both existing one-day temporal stages, six full-panel scores. This tests sensitivity to current capture selection before adding a new mechanism. Future captures differ, reused temporal stages are development, independent embryo pairing and 64-replicate certification remain unavailable.

All proposed specialist solutions remain unimplemented candidates; no claim that the problems are solved, no altered scorer, no automatic official upload.
