# Conditional TIGON full-panel metric reviews

Report hash 2e3756d7b592372ce3c30cc35c81e6f55b64436ece25b57e9c07aff5925751f6; specialist role version 2. Remedies are unvalidated and are not selected automatically.

## de_score

Problem: Both conditional TIGON variants have DE raw scores 0 and skills 0.5 across all three queries, matching copy; no candidate passes. CNF has positive DE scores but differs in architecture and training budget. This supports failure of the fixed TIGON configuration, not general TIGON failure. Near-uniform weights and tiny covariance changes suggest weak transport; causality remains uncertain. Previously exposed targets and absent embryo independence limit interpretation.
Proposed solution: Using only past windows, test stronger donor-residual transport through existing anchor heads; choose transport strength and growth/backoff settings without target tuning. Keep the full-feature scorer and calibration unchanged.
Validation: Freeze past-selected settings before replay. Compare copy, unchanged TIGON g0/g1, and revised g0/g1 with shared initialization, exact queries, and matched budgets; retain CNF as an unmatched reference. Success requires positive DE improvement over copy and unchanged TIGON across predeclared past folds without guard failures; otherwise reject. Fixes remain unvalidated; readiness unchanged.

## de_direction

Problem: Both conditional variants have DE-direction raw 0 and skill 0.5 across all reported outcomes, matching copy. This could reflect no called DE; call counts are unavailable. It does not establish correct-direction improvement. No candidate passes. Growth adds effectively no contrast. CNF differs in architecture and training budget, so its comparison cannot isolate conditioning effects.
Proposed solution: In a new past-only development cycle, fit a signed residual correction using existing past anchor heads and donor residuals. Select its strength on held-out past transitions, retain the original scorer/calibration, and freeze replay before future access. Do not tune against the exposed 8 target.
Validation: Compare correction on/off and growth on/off with shared initialization, exact queries, architecture, and matched training budgets, alongside copy. Report DE call coverage and signed agreement. Success requires reproducible direction improvement with adequate coverage on untouched evaluations; raw 0/skill 0.5 or unstable gains fail. Remedies remain unvalidated; readiness unchanged.

## mmd_u

Problem: MMD-U calibration is valid, but neither TIGON candidate passes. Both are slightly worse than copy in every reported raw replicate; CNF is better. Growth's mean skill improvement (~0.00000349) is mixed across replicates and unvalidated. CNF comparisons confound architecture and training budget; exposed-target retrospective results cannot establish embryo independence.
Proposed solution: Using only past data (<=7.75), investigate near-copy behavior and low covariance; select any diversity-preserving correction through past-only validation, then freeze it. Preserve original scoring, calibration, donor residuals, and replay guarantees. Treat this remedy as unvalidated.
Validation: Compare growth-enabled/disabled models with shared initialization, identical queries and budgets, copy controls, and budget-matched CNF controls. Success requires reproducible MMD-U improvement over copy and growth-disabled controls, uncertainty excluding no improvement, and the unchanged progress gate passing on an unexposed future target. Failure includes near-copy performance, inconsistent replicate gains, or broken replay. Readiness remains unchanged.

## variogram

Problem: All three variogram raw scores are worse than copy for both conditional variants; growth further worsens each raw score. Valid calibration supports a fixed-configuration failure, not blanket method rejection. Near-uniform donor weights and low covariance suggest attenuated dependence, but do not establish causality. CNF differs in budget and architecture; exposed-target retrospective results lack embryo independence.
Proposed solution: Using past-only rolling cutoffs, test sharper conditional donor weighting and a residual-covariance floor while retaining donor identities and residuals. Select and freeze settings before target evaluation; these remedies remain unvalidated.
Validation: Compare growth on/off with identical architecture, initialization, queries, training budget, and resampling seeds; include copy and budget-matched CNF controls. Keep the full scorer and calibration unchanged. Success requires reproducible variogram improvement over copy and a positive growth contrast across independent held-out embryos, with replay/freeze checks passing. Failure includes absent or unstable gains. Readiness remains unchanged.
