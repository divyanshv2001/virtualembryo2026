# Capture gene-mean shrinkage audit: four specialist reviews

Private report SHA256: a4442fb533cc37a693ba07dfc43f8914423d19fd42d4d456b2584aa9c031bbb1. Specialist policy version2. Fresh agents, compact metric packets; each review below is under150words. Actual benchmark metrics, skills and calibration are unavailable. Reward0; balance remains−663. Gene-mean MSE does not establish DE, direction, MMD or cell-state structure skill.

## DE

**Problem:** Capture-dispersion shrinkage has weak development support: it beats persistence in30/76 cases, versus37/76 for quarter-step and23/76 unshrunk. Paired MSE excess is+0.0002125/+0.0007583/−0.0001636 across cutoffs. These are gene-mean proxies; benchmark DE is unavailable, and captures are not established embryos.

**Proposed solution:** Retain equal-capture, zero-inclusive log means and eligibility thresholds. For a new candidate, tune gain caps and persistence fallback using only earlier eligible capture-stage transitions; freeze before evaluation. Preserve the three-fold results without retrospective tuning. Mark unsupported first-fold CM, cardio-pharyngeal and endothelial cases.

**Validation:** Compare persistence, quarter-step, unshrunk and cell-weighted controls on identical eligible cases and genes. Require lower paired proxy MSE across cutoffs and independently improved full-panel DE scoring; proxy gains cannot establish DE skill. Reject inconsistent gains or unsupported coverage. Readiness unchanged.

## Direction

**Problem:** Actual direction is unavailable. Mean-MSE proxies cannot establish direction improvement. Shrinkage beats persistence in30/76 cases, with positive mean excess at two stages. Captures are not independent embryos; first-fold CM/cardio-pharyngeal/endothelial support is absent.

**Proposed solution:** Retain equal-capture log means including zeros and thresholds. Compute deltas and variance-adjusted gains from past data only. Freeze a cell decoder and DE calling/sign thresholds fitted exclusively to earlier stages before forecasting; use persistence for unsupported lineages.

**Validation:** Compare persistence, quarter-step, unshrunk and shrinkage on identical supported cases across frozen folds. Report actual direction alongside full-panel scores, MSE, support and embryo-clustered uncertainty where identifiers exist. Require consistent direction gains without material panel degradation; failure includes inconsistent gains or unavailable decoding. Readiness unchanged.

## MMD

**Problem:** MMD and calibration are unavailable. Shrinkage improves gene-mean MSE over persistence in30/76 cases, with stagewise excesses+0.0002125,+0.0007583,−0.0001636; this does not establish distributional improvement. Captures are not independent embryos.

**Proposed solution:** Fit a past-only cell decoder preserving forecast lineage means while learning within-lineage variation. Freeze decoder choices and forecasts before evaluation. Compare matched persistence, quarter-step, unshrunk, shrunk and cell-weighted controls using identical cells, preprocessing, kernels and sampling budgets.

**Validation:** Run the existing full-panel MMD scorer without spatial inputs. Require valid calibration and consistent paired improvement across held-out stages, with embryo-level uncertainty where identifiers permit; otherwise report capture-level uncertainty. Failure includes unavailable calibration, inconsistent gains or merely reproducing anchor MMD. Readiness unchanged.

## CSS / variogram

**Problem:** Cell-state structure variogram is unavailable. Mean-only forecasts lack a cell decoder, so MSE cannot establish structure preservation. Shrinkage beats persistence in30/76 cases; excess MSE changes sign across stages. Captures are not embryo replicates.

**Proposed solution:** Fit a regularized past-only cell-state residual decoder, freeze before evaluation. Preserve candidate lineage means while transporting residual structure in expression space. Select covariance regularization using past-only capture holdouts; the prior1/7 advantage over persistence warrants caution.

**Validation:** Use the full-panel scorer with identical cells, genes, lineage eligibility, decoder and mean shifts across persistence, quarter-linear, unshrunk and shrunk controls. Include a mean-preserving tangent control near the anchor. Require lower held-out nonspatial variogram error with capture-level uncertainty; missing scores or unstable gains constitute failure. Readiness unchanged.

## Coordinator decision and limits

The pinned scorer, calibration and DE calling thresholds stay unchanged; the direction critic's threshold suggestion is rejected where it would alter scoring. Assay-capture uncertainty must not be represented as embryo uncertainty. Unsupported cases are omitted with explicit reasons in this proxy audit; any later full-panel decoder must use persistence for unsupported lineages without dropping scored cells or genes. The current audit uses unconstrained log-mean vectors, not valid nonnegative cell forecasts. Feasibility and exact row-mass/protected-gene guards must be resolved before full-panel testing.

Jev selected a fixed decoder preflight (923input/60output tokens, confidence0.99); no gain-cap search on these exposed targets. The earlier estimator decision used891input/59output tokens. A local packet-writing syntax error occurred before any synthesis request; it was repaired, with one actual advisory request for that decision. No measured Codex-credit savings are claimed.
