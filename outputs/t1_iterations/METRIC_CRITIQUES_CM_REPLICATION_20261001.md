# Fixed CM .25 capture replication: specialist review

Batch cm_capture_replication_fullpanel_01, role-policy version2. Six valid full-panel scores. Current captures33/24 were excluded from all base fits; their CURRENT observations calibrated. Targets are different captures at reused development stages, not independent paired embryos. No candidate passes. Anchor49.924356/50.118393;CM.25 49.922955/50.087033;copy50/50. Separate reward -475 to -482. No upload.

## DE recovery

Problem: CM DE skills .464956/.441739 are below copy.5 and anchors.465539/.442895. Aggregate evidence cannot identify incorrect signs versus ranks. The critic requested checking headline interpretation: headlines above50 are explained by the unchanged weighted metric tradeoff, not evidence of an aggregation error or a recovered hidden target property.

Proposed solution: learn gene-wise temporal effect reliability from earlier rolling cutoffs and shrink unreliable effects toward copy, freeze thresholds before testing. This overlaps earlier reliability/EB/count-shrinkage failures and is not automatically a distinct viable implementation.

Validation: matched remedy/copy/anchor/CM.25, same full-panel scorer/splits; require DE gains on both folds, no target rank fitting or score inversion. Failure rejects the setting, not the entire family.

## Direction

Problem: CM direction beats copy but improves anchor on only the first fold (+.001556 raw), regressing on the second (-.000559). Incremental benefit did not replicate.

Proposed solution: past-only sign-reliability gating and shrinkage, suppressing uncertain corrections; freeze before target reads. This also overlaps earlier failed sign gates and needs a distinct specification before execution.

Validation: matched anchor/copy/CM.25 plus magnitude-matched shuffled corrections; positive gains on both folds beyond prespecified uncertainty. Cell resampling does not establish embryo replication.

## MMD

Problem: CM raw MMD .249096/.213013 improves copy .266957/.232818, but barely changes anchor: -.0000668 then+.0000325. This supports the base anchoring, not the extra CM offset.

Proposed solution: choose observation shrinkage on strictly earlier rolling cutoffs then freeze. Existing strength trials constrain this proposal; no further tuning on exposed future targets is selected here.

Validation: identical cells, preprocessing, kernels, sample budgets and seeds; require paired raw-MMD improvement over both controls on each fold with capture-level uncertainty.

## Nonspatial CSS

Problem: CM raw CSS .00952891/.00793751 regresses versus copy/anchor in the first fold and improves the second. Tiny mixed differences do not establish structural benefit.

Proposed solution: past-only covariance/variogram residual correction with shrinkage, current anchors for conditioning and heldcapture exclusion from base. This is speculative and must preserve the original scorer.

Validation: same observations/features/pair weights; require lower raw CSS than copy and anchor on both folds. Regression or future-dependent tuning rejects the method.

## Next distinct bounded test

Jev chose mean_mass_projection_feasibility (confidence1.0), one2149-byte request, actual906input/59output tokens. Existing coupling preserved margins before raw row-mass repair but did not jointly enforce log-gene means and raw row masses. Test a fixed200-iteration alternating projection on CURRENT CM observations only, locked post-switch support, tolerances1e-5, across the four already-declared observed captures. Current-only synthetic odds perturbations are numerical feasibility stress tests, not temporal forecasts. Explicit impossible support/nonconvergence stays invalid; no fallback is counted as success. Even matching means cannot guarantee Mann-Whitney DE ranks. No headline/raw4/skills4 or reward is claimed for this feasibility audit. Only if all cases converge may a new temporal test be predeclared; other structural/population families remain open.
