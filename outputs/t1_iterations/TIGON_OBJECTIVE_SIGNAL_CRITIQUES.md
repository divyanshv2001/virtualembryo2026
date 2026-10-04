# Past-only TIGON objective-signal critiques

Report hash e0963acdc8b33bda3d1bbe8f933d96c830615d81324938ec0d44dad334b69ff5; role version2. Benchmark metrics unavailable; zero reward. Supplied compact diagnostic evidence is preserved in TIGON_OBJECTIVE_SIGNAL_RESULTS.json. Remedies unvalidated.

## de_score

Problem: DE score is unavailable. Density gradients are much smaller than energy gradients, and optimization approaches zero displacement. Final original-objective loss slightly exceeds the zero-field control. This supports weak density influence, but establishes neither causality nor a global optimum.
Proposed solution: Test gradient-balanced loss weighting: calibrate a density multiplier using initial density/energy gradient norms from frozen past-only queries, then hold it constant throughout optimization. This changes objective weighting without introducing signed drift or another fixed conditional configuration.
Validation: Match encoder, queries, initialization, optimizer, steps and seeds across baseline, calibrated weighting, and common-loss-rescaling controls. Predeclare success as reproducible density-loss reduction of at least 1%, energy no higher than its initial value, and original-objective loss below zero-field loss. Failure includes instability, improvement explained by common rescaling, or continued collapse toward zero displacement. Benchmark benefit remains unverified; readiness is unchanged.

## de_direction

Problem: On one frozen past-only query, energy gradients dominate initially, and optimization approaches zero transport despite little density improvement. This supports an objective imbalance diagnosis, not causal or global proof. DE-direction metrics remain unavailable.
Proposed solution: Test a past-only, density-first constrained optimizer: require density loss to improve over the zero-field baseline by a margin estimated from repeated past-only noise draws; minimize energy only within that feasible region. Estimate the margin without target access. This changes the optimization rule rather than repeating signed drift or fixed-weight configurations.
Validation: Compare against the unchanged objective and zero field using identical initialization, query/noise streams, and enforced evaluation budgets; historical call counts are unavailable. Pass diagnostically only if held-out past-only draws retain density improvement with bounded energy and reproducible nonzero transport. Fail if infeasible, unstable, or effectively zero. Benchmark success requires subsequently available DE-direction metrics; readiness is unchanged.

## mmd_u

Problem: Benchmark MMD is unavailable. The past-only diagnostic supports weak density gradients under limited overlap: energy shrinks sharply while density barely changes; the final objective slightly exceeds zero-field. This does not establish a global or causal failure.
Proposed solution: Test energy-constrained density descent: use the unchanged density gradient to propose per-query trust-region steps, project onto a past-calibrated kinetic-energy budget, and backtrack against the unchanged total objective. Freeze encoder, models, kernel, and scorer. Select budget and radius using past-only splits; do not repeat failed fixed configurations or tune against targets.
Validation: Match queries, initialization, compute budget, and seeds against the existing optimizer and zero-field control. Require reproducible density reduction with total objective below both controls, finite gradients, and stable ESS. Reject if improvement depends on one split, violates the energy budget, or merely reduces energy. Benchmark MMD remains unavailable; readiness is unchanged.

## variogram

Problem: The one-query diagnostic shows weak density gradients and near-zero-field preference, consistent with the pilot's negligible covariance change. It does not establish global causality. Variogram scores are unavailable.
Proposed solution: Test a bounded affine deformation field centered on the past centroid, using a covariance-evolution loss estimated exclusively from past observations. Normalize covariance residuals and regularization using past-only scales, then freeze them before evaluation. This directly tests whether an identifiable deformation signal avoids collapse.
Validation: Use identical queries, donors, integration settings, and compute budgets for the unchanged objective, zero field, centroid translation, and proposed deformation. On withheld past intervals, require improved covariance prediction and noncollapsed deformation without unstable energy or donor concentration. Reject if improvements disappear against matched controls or depend on retrospective tuning. Preserve the original 32-query component-equality check. No target access or repeated failed fixed configurations. Readiness remains unchanged.
