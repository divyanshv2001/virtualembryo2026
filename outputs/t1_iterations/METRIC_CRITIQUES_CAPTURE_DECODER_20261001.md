# Capture-mean decoder feasibility: specialist reviews

Private report SHA256 dc28b09682f8ba20de9db3c157486d5492c21d5b9ab599395d61b6b0614d6617; role policy2. Four fresh-context specialists, <=150words each. No target expression read, no benchmark metrics/skills/calibration/reward. All six donor-only cases passed the predefined guards; numerical feasibility is not predictive success.

## DE

**Problem:** All guards establish numerical safety, not DE fidelity. Mass repair substantially distorts intended lineage shifts (relative error0.232–0.491), and the proxy audit shows no consistent improvement over persistence, especially in the first two folds.

**Proposed solution:** Keep the decoder experimental. Using past donors only, replace post-hoc mass repair with a constrained mapped-count projection preserving row mass while minimizing deviation from intended lineage log-expression shifts. Freeze before evaluation; leave scorer and calibration unchanged.

**Validation:** Compare with matched-donor copy and incumbent controls on identical past-only held-out folds. Require all guards, improved lineage-shift fidelity and aggregate proxy improvement over both controls without worsening the first two folds. Actual DE scoring remains necessary. Current evidence supports preflight safety only.

## Direction

**Problem:** Safety passes, but direction fidelity remains imperfect: quarter shifts achieve cosine0.917–0.974 and relative error0.232–0.408; variance-shrunk shifts perform worse. Prior proxies do not establish benefit over persistence. Capture comparisons cannot establish embryo generalization.

**Proposed solution:** Keep scorer calibration/thresholds unchanged. Select strength using completed past-only folds, with donor/column matching fixed across persistence, incumbent and decoder candidates. Preserve unsupported lineages and protected genes.

**Validation:** In held-out historical folds, compare decoded versus observed lineage changes using cosine, relative error and signed agreement on prespecified supported genes. Require improvement over both controls consistently across folds, retaining guards. Report capture-level resampling uncertainty; reserve embryo claims for independent matched embryos. Current evidence supports safe execution, not predictive direction.

## MMD

**Problem:** Guard compliance establishes feasibility, not improved MMD. No MMD was evaluated; captures cannot be treated as independent embryos. Earlier proxies were mixed, with worse mean errors in the first two folds.

**Proposed solution:** Freeze quarter and shrunk variants using past donors/capture means. Compare with persistence, incumbent and matched-donor copy using identical donor selection and evaluation populations. Use no spatial inputs.

**Validation:** Run unchanged full scorer/calibration on held-out future captures without tuning on expressions/scores. Report paired MMD differences by cutoff and available independent embryo, alongside guards. Require reproducible improvement over incumbent and copy, preserved guards and no material degradation in other metrics. Failure includes inconsistent gains, pseudoreplication or guard violations. Retain incumbent if neither variant passes.

## CSS / variogram

**Problem:** Six guards establish decoder integrity, not improved cell-state structure. Intended lineage changes are approximately recovered; prior proxies favored persistence on many comparisons. Captures are evaluation units, not embryos.

**Proposed solution:** Freeze cut8/8.25, means, scorer and calibration. Compare quarter/shrunk updates against incumbent and matched-donor copy using identical matching/protected-gene handling. Select strength using past captures only; add no spatial inputs.

**Validation:** Run unchanged scorer on complete32285-gene panel. Require actual CSS improvement and reduced nonspatial variogram discrepancy relative to both controls across held-out captures, with capture uncertainty and no systematic fold degradation. Retain guards. Current mass error, mean shift, covariance change and lineage alignment support feasibility only; CSS remains unmeasured.

## Decision

No strength selection on already exposed proxy targets. Constrained projection is a possible future method, not assumed feasible: earlier joint mean/mass solvers were infeasible or near-anchor. Jev chose actual full-panel testing of these two fixed decoder variants (950input/49output, confidence1.0), rather than another projection proxy. The decoder design advisory used800input/61output, confidence0.41. Scorer/calibration unchanged; matched copy and archived incumbent must be included. Passing guards does not meet readiness or justify a new submission. The next scoring run must regenerate and verify exact preflight prediction hashes before reading targets.
