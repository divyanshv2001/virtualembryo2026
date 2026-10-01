# Scope audit specialist critiques — policy version2

Private report SHA256: 37756100392093c722577f7f0a3906af7600d7100df893505c2d4eac64acf455. Four fresh-context agents, <=150words each; summaries below. Captures13 and12 are identifiers, not replay counts. Audit has no benchmark metrics or reward.

## DE score

Problem: No DE score available. NonCM changes confounded legacy mean shifts. Proposed solution: Use an explicitly CM-only anchor operator; separately learn past gene-pseudobulk corrections if targeting DE because exact mean preservation limits this metric. Validation: Hash-exact controls, no future tuning, real unchanged-scoring gain over anchor/shuffle; unavailable scores imply no promotion.

## Direction

Problem: Scope compliance establishes no directional gain; previous learned/shuffled results indistinguishable. Proposed solution: Freeze anchor-based operator, use past reference-residualized gene deltas with shuffled/sign-reversed controls. Validation: Reproducible partial-Spearman gain, exact protected cells and valid constraints. Missing metrics, equivalent controls or invariant violations fail.

## MMD

Problem: Legacy changes ~3.2million nonCM entries; no audit MMD. Proposed solution: Frozen anchor-base/copy/identity/shuffle comparison with matched norm, support, mass, covariance and backoff. Validation: NonCM bitwise equality first, then actual paired MMD benefit beyond matched controls; no future tuning. Readiness unchanged.

## CSS/variogram

Problem: No variogram metrics; CSS measures cell-state structure, not spatial structure. Proposed solution: Frozen anchor-base matched controls and past-only fitting, preserve nonCM bitwise and match perturbation norm/constraints. Validation: Actual reproducible CSS benefit over controls; unavailable metrics or learned-shuffle equivalence preclude promotion.

Coordinator: Wholepanel shifts .005837/.011640 are dominated by nonCM contribution; CM contribution ~2e-9. Anchor-base errors2.1e-9/1.7e-9, nonCMchanged0. Bothbackoff1; cumulative systemic switching plausible mechanism, not uniquely attributed. Proposals altering gene means conflict with mean-preserving operator and require a separate declaration. Jev1929bytes812in63out selected frozen isolated-scope full-panel diagnostic. Reused targets, no independent promotion.
