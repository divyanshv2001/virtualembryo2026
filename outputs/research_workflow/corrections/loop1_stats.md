# Critic loop 1 correction: C1-2

**Disposition:** accept the algebraic and decision-rule criticism. This correction supplies a prospective statistical contract; it reports no biological outcome and assigns no numerical scientific margin. The original formula in `EXPERIMENT_DESIGN.md:35` conflates a success threshold with the distance to the planning alternative. Parent integration into the design and synthesis remains required.

For the provisional primary released-stage T2 embryo interpolation proxy, let

`D_i = mmd_u(composition-only, target embryo i) − mmd_u(combined, target embryo i)`.

Both models use the frozen sampling/refit recipe and are evaluated against the same target embryo. Average conditional numerical repetitions according to the frozen recipe before obtaining the embryo contrast; repetitions do not become independent biological observations. Let `μ = E[D_i]` under the declared target-embryo population and weighting scheme. Freeze the task/horizon, scorer contract, recipe, independent assessment manifest and practically meaningful margin `δ* > 0` before accessing assessment outcomes.

The confirmatory superiority question is `H0: μ ≤ δ*` against `H1: μ > δ*`. Success requires a valid one-sided lower confidence bound `L_(1−α)` strictly above `δ*`. Merely observing `D > 0`, or rejecting a zero-effect null, does not establish practical superiority. Persistence remains a secondary comparator unless a separate, preregistered superiority requirement is added. No claim about other tasks, official hidden targets or extrapolation follows from this proxy.

For independent embryo contrasts with planning standard deviation `σ_D`, anticipated mean `μ_A > δ*`, and desired power `1−β`, the normal approximation is

`n ≈ [(z_(1−α) + z_(1−β)) σ_D / (μ_A − δ*)]^2`.

Here `δ*` is the required benefit, `μ_A` the anticipated alternative, and their difference the distance the test must resolve. This expression assumes independent contrasts and an adequate normal approximation; it is neither an exact small-sample calculation nor a cluster-design formula.

**Purely mathematical illustration, not an embryo result or recommended margin:** hold variance and critical values fixed. Let the alternative initially be `μ_A = δ* + h`, where `h > 0`. Moving it to `μ_A = δ* + h/2` makes the approximate required sample size four times larger. In general, `n ∝ 1/h²`, so planning replication diverges as the anticipated effect approaches the success threshold. No observed variance or numerical n exists here.

If secondary endpoints support a joint “no deterioration beyond tolerance” claim, define each `G_ij` so that positive values mean worse combined-model behavior. For a higher-is-better direction endpoint use comparator minus combined; for lower-is-better variogram or neighborhood discrepancy use combined minus comparator. Apply absolute deviation before differencing signed ideal-zero scale/severity endpoints if that is the frozen endpoint contract. Define scientifically justified allowed losses `η_j` before assessment, and require simultaneous upper confidence bounds `U_j < η_j` for every prespecified guardrail. Use a preregistered family-wise simultaneous procedure, such as Bonferroni upper bounds, when claiming joint coverage. Include all applicable direction, co-expression and T2 spatial guardrails; report other official components and the official weighted composite separately. These endpoint tolerances cannot be chosen from the final target. A simpler defensible option is to make every secondary endpoint descriptive and remove the no-degradation claim. Nonsignificant harm alone is insufficient in either case.

Pairing refers to two model scores on the same target embryo, not matched cells or the same embryo followed through destructive stage assays. Embryos sharing a litter or processing batch may be dependent, requiring a justified cluster-level analysis; embryo count is not automatically the effective independent n. Small n makes variance, tail coverage and bootstrap reliability uncertain. Use small-sample-aware inference or design simulations only when the manifest and replication support them. With one embryo per stage, stage and embryo are confounded: report observed D, all component changes, and conditional sampling/refit sensitivity descriptively, without population confidence, confirmatory success, or a power claim.

**Remaining empirical risks:** no eligible pilot variance, fixed numerical margins, independent sample manifest, scorer implementation, or demonstrated target-domain transportability is available. Cluster structure and nonlinear stage effects remain unknown. Residual-based selection on assessment outcomes consumes that assessment; these formulas cannot restore independence. Failure to clear the margin is inconclusive unless the uncertainty actually contradicts the claimed benefit; absent evidence must remain untested.
