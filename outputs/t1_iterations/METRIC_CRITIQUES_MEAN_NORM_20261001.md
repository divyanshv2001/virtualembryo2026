# Mean and displacement feasibility critiques

Private report SHA256: `2ab85d136c4f61c068ff82df7fa8824c207d875a4e387a84dda06610091da163`. Specialist role-policy version: 2. Four fresh agent contexts; no new outcome metrics or reward in this donor-only audit.

## DE specialist

**Problem:** Exact additive quarter/shrunk mean matching is infeasible under locked zero support: 4,504/4,070 unsupported gene–lineage pairs occur at cuts 8/8.25. Their mean mass is tiny, but supported minimum energy exceeds actual quarter energy by 1.526/1.516× and shrunk energy by 3.927/3.954×. Anchor matching also has 355/414 unsupported pairs. Copy/blocked passing this necessary bound does not establish mass or nonnegativity feasibility; prior blocked DE results did not pass.

**Proposed solution:** Use past-donor-only support projection, then damp supported mean shifts within the existing perturbation-energy budget. Enforce mass and nonnegativity explicitly. Compare copy and blocked controls without a target grid, future expression, new metrics, or reward changes.

**Validation:** Require exact replay of all 12 forecast hashes. Audit the necessary bound before forecasting, then assess existing DE and CSS criteria at both cuts. Accept only joint improvement; better early CSS with worse direction is insufficient. Treat captures as captures, not independent embryos.

## Direction specialist

**Problem:** Donor-only updates cannot reproduce nonzero decoded mean changes where donor support is zero. Even supported directions exceed the quarter-step energy budget by 1.516–1.526×, and shrunk budgets by 3.927–3.954×. Small unsupported energy fractions do not establish feasibility; blocked/copy passes do not prove nonnegative mass feasibility.

**Proposed solution:** Using past data only, project the decoded mean-change direction onto donor-supported, nonnegative updates under the existing energy budget. Minimize deviation from that direction; permit attenuation rather than requiring incompatible exact mean and norm matches. Keep donor zeros locked and use copy as the explicit fallback.

**Validation:** Run the same exact 12-forecast replay at both cuts, with additive-quarter, shrunk, blocked, and copy controls. Accept only if existing acceptance criteria pass at both cuts and support, nonnegativity, and budget constraints hold. Treat reused development captures as development evidence. No target reads, new reward, or new outcome metrics.

## MMD specialist

**Problem:** Zero-lock makes exact decoded mean and Frobenius-energy matching impossible for quarter and shrunk fields: supported minimum energy exceeds actual energy. Tiny unsupported mean fractions do not remove this obstruction. Copy/blocked necessary checks do not establish row-mass or nonnegativity feasibility.

**Proposed solution:** Project the blocked field onto each donor row’s nonnegative, donor-positive-support simplex, preserving donor row mass. Interpolate from donor to this projection using one amplitude fixed from past captures. Construct controls through the same projection and interpolation; require a common attainable energy and permit residual mean mismatch. Do not claim exact mean matching or expose a target-selected amplitude grid.

**Validation:** Replay all 12 forecast hashes exactly. Verify support, nonnegativity, row mass, attained energy, and mean residuals before interpreting existing MMD and forecast scores. Require the existing two-fold gate; prior blocked scores fail it. Early CSS improvement alone cannot override worse direction. Treat captures as observations, not independent embryos.

## Cell-state structure specialist

**Problem:** Zero-lock makes 4,504/4,070 desired gene–lineage mean shifts unsupported at cuts 8/8.25. Even supported shifts require more minimum energy than quarter or shrunk controls provide. Tiny mean-energy fractions do not establish feasibility; early zero-block CSS gains remain below the 49.46 anchor.

**Proposed solution:** Preserve donor-positive support and nonnegativity, but replace exact requested shifts with feasible projected shifts. Solve constrained minimum-energy adjustments, then retain the largest adjustment compatible with the existing energy budget. Keep the anchor when CSS or direction deteriorates. Copy/blocked checks are necessary controls, not feasibility certificates.

**Validation:** Use only past-donor controls on the same 12 scored forecast hashes. Report unsupported pairs, achieved decoded mean shifts, nonnegativity violations, energy, CSS, direction, and variogram deviation against anchor and zero-block controls. Require both cuts to pass existing criteria. Treat captures as repeated observations, not independent embryos; use no future expression, spatial inputs, or new benchmark.

## Coordinator interpretation and next decision

The DE phrase “mean mass” means squared gene-mean-shift energy, not count-library mass. Frobenius displacement is measured in log-expression coordinates; row count mass is measured after expm1. Neither necessary support/energy bounds nor a count-space simplex projection proves their simultaneous feasibility. CSS concerns cell-state structure, with no spatial input established.

These are proposals, not certified solutions. All 12 original hashes already replayed exactly. New diagnostic forecasts necessarily have new hashes and must preserve original control hashes. The CSS suggestion to retain an anchor when metrics deteriorate cannot become target-selected fallback: any fallback rule must be frozen from past-only diagnostics. New full-panel validation, if separately declared, would use all four unchanged metrics and the original reward policy; this audit itself has zero new metrics/reward.

Jev synthesis selected `supported_mean_energy_preflight` (reported confidence 1.0; advisory only), 879 input / 58 output tokens, 2,379-byte packet. Next: construct the analytic minimum-L2 donor-supported mean field, attenuate to the frozen additive displacement budget, then audit clipping and count-mass repair. Test whether it differs materially from the already tested blocked decoder. No target expression, strength grid, or promotion from this preflight. Exact matching is not promised; families and readiness remain open.
