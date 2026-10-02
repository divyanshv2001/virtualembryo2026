# scNODE past-fit audit metric critiques

Four fresh methodological specialists, role-policy version2. No benchmark metrics, score, reward or held-out validation. Reconstruction diagnostics use existing <=8.5-fitted models and E8.5 anchors.

## DE

Problem: DE unavailable. Variance loss/zero-variance genes can coexist with high gene-expression correlation; no DE fidelity claim. ReLUs and Euler mismatch are hypotheses.
Proposed solution: Frozen checkpoint Euler1/.25/.125day comparison, same decoder/cells/latents; inspect gene variance/zeros/changes. Existing anchored residual is already present.
Validation: Source/anchor/zero-step/currentstep controls. Require numerical convergence; otherwise reject integration as a remedy. Honest earlierfolds need fresh cutoff fits.

## Direction

Problem: Direction unavailable. Variance contraction and large latent movement justify diagnosis, not predictive attribution.
Proposed solution: Frozen models with one,two,four,eight substeps, zero-drift/zero-step, inspect trajectories/preactivations/zero-variance genes.
Validation: Matched past cells/controls and numerical convergence. Proposed reconstruction improvement cannot be caused at timezero by substepping; future validity still requires fresh earliercutoff models.

## MMD

Problem: No MMD-U, no distribution improvement. Low anchor variance, large shifts and clipped outputs suggest contraction without proven source.
Proposed solution: Frozen one-day/quarter/eighth-step rollouts, identical starting latents and decoder; inspect displacement/variance/zero fractions/changes.
Validation: Zero-time/currentstep controls and finer-step convergence. Persistent contraction favors representation/decoder investigation; fixed benchmark scorer only after separatelyvalidated recipe.

## Variogram

Problem: Benchmark unavailable, no coordinates/spatial claim. Low variance and possible ReLU/time-step mismatch need separation.
Proposed solution: Frozen quarterday/finersteps, sameelapsedtime/cells/preprocessing, log displacement/decodedvariance/saturation, no new anchor.
Validation: Numerical convergence and stable proxies against identity/current controls; any biologicalgain needs honest cutoff retraining.

Coordinator corrections: Beta.1 source8.5 mean variance retention is11.4%; E8.5 anchor is4.8%. Zero-step identity means decoded(z+0*drift)==decoded(z), not decoded(z)==observed expression. It does not establish dynamics as cause of reconstruction loss. A finer solver cannot repair timezero reconstruction; retain this defect as a separate hypothesis. No benchmarkscore or official rootcause attribution. Jev826input/59output,1665bytes selected one Euler convergence audit. Past/source checkpoint and same-training-draw provenance verified. No tuning on9.5/hidden10.5 or automatic TIGON.

Private report SHA256: b4b5701399a00f45e5dee9ff2a19383195985574c43bdf26f224ed8c2f193985
