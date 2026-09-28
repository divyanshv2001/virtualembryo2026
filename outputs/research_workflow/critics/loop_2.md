# Chief Scientific Critic — loop 2 of exactly two (final)

This is the second and final independent critic review. The scoped audit and prospective protocol can finalize after integrating the concrete documentary corrections below and explicitly retaining execution gates. They cannot finalize as a validated predictor, confirmatory-ready experiment, eligible competition design or novelty claim. No biological datasets were accessed, no prediction experiment was run, and no Jev request was made. Loop-1 corrections improved scope and power logic; design fixes are not biological support.

I reread S1, revised design, resolution and specialist corrections, exact arithmetic-check source, scorer/exposure/closest-work contracts, preregistration and current registries. The most material remaining error is a mathematically incorrect equal-embryo mixture claim. The second is attribution: this is endpoint-mixture recentering, not isolated composition-versus-state prediction.

## C2-1 — Major — owner: genmodel

**Flaw:** The donor sampler is not the claimed convex mixture of equal-embryo endpoint empirical distributions.

**Why it matters:** This changes the baseline distribution and its conditional means, so the primary comparison and advertised embryo weighting are mathematically inconsistent.

**Evidence:** corrections/loop1_genmodel.md averages proportions p_sk across embryos, then samples uniformly among embryos containing k. Let two equal-sized endpoint embryos have k proportions .9 and .1. p_sk=.5. The proposed sampler assigns joint mass .25 to each embryo within k, whereas an equal-embryo empirical population assigns .45 and .05. It redistributes rare-state donors. This failure occurs before biological uncertainty or output normalization.

**Required fix:** For endpoint s and stratum k choose embryo e with probability p_esk/sum_e p_esk, then a uniform k row. Use mu_sk=sum_e p_esk*mu_esk/sum_e p_esk. Equivalently draw endpoint, embryo uniformly, then a row uniformly before identifying its stratum. If retaining uniform-within-k embryos, explicitly define a different pseudo-population, cease claiming an equal-embryo endpoint joint mixture and explain why that estimand is appropriate. Correct all dependent baseline descriptions.

**Verification:** Add a deterministic multi-embryo unequal-proportion example that verifies exact endpoint row masses, stratum means and .6/.4 mixture identity. Include absent strata and unequal capture sizes. Current checks only verify pi/q sums and cannot catch this error.

**Residual risk:** Correct weighting cannot identify biological abundance under differential capture. Small strata may still be noisy; observed specimen identity may not imply independent embryo identity.

**Status:** open at final critic review. Subsequent integration may resolve documentation without a third critic pass or empirical support.

## C2-2 — Major — owner: benchdesign/genmodel

**Flaw:** The comparator already interpolates conditional expression distributions, and the shifted estimator changes between-endpoint variance as well as conditional means.

**Why it matters:** Calling this composition-only versus conditional-state benefit attributes a distributional gain to the wrong biological contrast. Both models contain temporal state information. A gain may arise from shrinking an artificial bimodal mixture, even with no change in the overall conditional mean.

**Evidence:** Composition-only samples actual A and B expression rows. If p_Ak=p_Bk, q_A=.6 and q_B=.4; its conditional mean is already mu_tk=.6mu_Ak+.4mu_Bk. The lambda=.5 shift leaves this mean unchanged but halves separation of endpoint centers. Its between-center covariance becomes (1−lambda)^2 q_A q_B (mu_A−mu_B)(mu_A−mu_B)^T, one quarter of baseline for lambda=.5, before nonlinear F. With unequal proportions, a mean change also occurs. This follows directly from the revised formula.

**Required fix:** Rename comparator endpoint-mixture and treatment shrunk endpoint-recentering. Narrow the primary claim to whether this frozen transformation improves the released-stage proxy. Do not claim isolated composition/state error attribution. A genuine composition-only factorial would require separately prespecified common conditional distributions and weights, with a new design and untouched assessment; it is not necessary for finishing this audit.

**Verification:** Provide a symbolic equal-proportion example showing unchanged conditional mean and reduced between-endpoint covariance, plus an unequal-proportion case. Final hypothesis, D labels, preregistration and synthesis use the narrow recentering contrast.

**Residual risk:** Nonlinear output mapping further changes means and covariance. Even a future positive D would establish predictive usefulness of this transformation, not causal state dynamics or a unique decomposition.

**Status:** open at final critic review. Subsequent integration may resolve documentation without a third critic pass or empirical support.

## C2-3 — Major — owner: spatial3d/genmodel

**Flaw:** The full joint mixture can combine unrelated embryo coordinate frames; retained row pairing does not make pooled neighborhoods meaningful.

**Why it matters:** Mixing individually meaningful clouds in separate frames can create spurious proximity, duplicate anatomy and invalid scale. Expression primary MMD can remain valid while descriptive T2 joint/spatial components become misleading.

**Evidence:** The recipe copies C_sj without registration and mixes endpoint embryos/stages. EXPERIMENT_DESIGN admits incompatible frames but still describes a convex endpoint joint mixture and official spatial components. SCORER_CONTRACT leaves frame/units null. Different embryos can have independent origins and orientations even at one endpoint.

**Required fix:** Set spatial assessment unavailable unless training-derived, documented common frame, physical units and coverage make pooled coordinates comparable; never derive registration from E7.25 truth. Keep expression-only primary analysis independent of this spatial gate. If no valid frame exists, preserve donor coordinates as a serialization placeholder explicitly excluded from anatomical/joint/neighborhood claims, or define per-embryo spatial scoring with its own estimand before access.

**Verification:** Final protocol has separate expression and spatial readiness flags. A synthetic pair of translated copies illustrates why row pairing passes while pooled neighborhoods change; do not imply desired scorer invariance without code. No descriptive spatial result is reported as interpretable until its frame contract passes.

**Residual risk:** Even verified frames leave capture coverage, anatomical correspondence, duplicate donor points, shape changes and topology unresolved.

**Status:** open at final critic review. Subsequent integration may resolve documentation without a third critic pass or empirical support.

## C2-4 — Moderate — owner: genmodel

**Flaw:** Output-map validity and annotation readiness are gated but still omit operational definitions that can silently redefine the estimator.

**Why it matters:** Panel closure normalization can invent relative expression changes if the original library denominator includes unobserved genes; exact-string strata may combine incompatible annotation protocols. UNKNOWN must not silently turn unavailable labels into a harmonized biological state.

**Evidence:** F normalizes sum(exp(x)−1) across the model panel to L; this round-trips only if that panel equals the normalization denominator domain. The future task panel/recipe are absent. The recipe says empty labels map to UNKNOWN but labels unavailable halt; no threshold distinguishes partial annotation from missing annotation. Strata use exact strings without documented annotation provenance.

**Required fix:** Specify that panel closure, denominator domain, log base, L and zero-row policy must be documented and jointly verified. If normalization was across a wider panel, stop this F route or carry a separately justified eligible denominator; never infer it from target. Record annotation method/version at each endpoint, reserved-token collision policy and allowed missing-label mass before fitting. A failed comparable-label contract stops this stratified recipe or invokes a separately frozen different analysis.

**Verification:** Add a small nonclosed-panel example that fails the normalization gate, an UNKNOWN-name collision case and inconsistent annotation-source example. Report before/after shifts, clipping and zero-row mass; no test on arbitrary vectors certifies real assay transforms.

**Residual risk:** Known annotation provenance does not guarantee identical biological semantics; sum normalization can erase total RNA changes and distort covariance.

**Status:** open at final critic review. Subsequent integration may resolve documentation without a third critic pass or empirical support.

## C2-5 — Major — owner: benchdesign/causal

**Flaw:** The synthetic access check does not exercise refusal or durable enforcement, although revised summaries can sound as though an access-control test passed.

**Why it matters:** A manually assigned boolean cannot prevent assessment reuse, altered lock files or exposed-context retrieval. A narrated ledger is an illustrative policy, not an implemented evaluator or clean-context boundary.

**Evidence:** baseline_contract_checks.py appends an access, sets confirmatory_available=False and checks not that flag; it never calls a reject-reuse operation, persists a ledger or attempts an excluded read. preregistration_template.json contains hand-written synthetic rejected events. causal correction correctly states launch restrictions/testing are future, not performed.

**Required fix:** Label these artifacts policy/arithmetic illustrations only and mark target-access enforcement and clean-context isolation not implemented/not tested. Keep execution gated until a durable lock/ledger rejects reuse against immutable model/data/config hashes and excluded reads actually fail in a future isolated launch. No new production harness is necessary to finalize this source-preserving audit.

**Verification:** Final readiness report distinguishes policy specified, illustrative event sequence present, implementation absent and enforcement untested. If later built, replay actual repeat-assessment/change-lock/retrieve-exposed-artifact calls and verify rejection. Do not call the present boolean assertion an operational boundary test.

**Residual risk:** Current exposed agents cannot be cleaned by a manifest; historical influence and pretraining exposure can remain unverifiable even with later isolation.

**Status:** open at final critic review. Subsequent integration may resolve documentation without a third critic pass or empirical support.

## C2-6 — Moderate — owner: sysbio/orchestrator

**Flaw:** Registry status and preregistration schemas still carry semantic inconsistencies after the prose correction.

**Why it matters:** The exclusive scientific surface should not confuse an observed code defect, an untested predictive hypothesis and a proposal disposition. A freeze gate requiring guardrail tolerances despite a descriptive-only decision also leaves future readiness ambiguous.

**Evidence:** hypothesis_registry.json labels all proposals untested and local_ground_truth=no, including sysbio-r1-1's observed legacy verdict-mapping contract, while E070 records it as code ground truth. All claim-scope booleans inherit board-based classification with warnings rather than reviewed explicit scope. preregistration freeze_gate requires all fields including eta_j/simultaneous-claim fields although revised EXPERIMENT_DESIGN drops guardrail claims. Mutable builder IDs/status preservation remain unrepaired, which is documented.

**Required fix:** Separate proposal/forecast status from evidence kind and logical/code finding status; retain biological outcomes untested while linking observed contract defects as observed code evidence. Use scope unknown/requires-review rather than asserting board-derived false where unreviewed. Mark conditional/not-applicable guardrail fields and define readiness predicates for descriptive versus confirmatory expression and spatial analysis. Preserve original-source defects as future repairs rather than pretending registry rebuilds enforce durable lifecycle.

**Verification:** A schema consistency review confirms no code finding is represented as biological falsification, no proposal confidence as calibration, and no optional guardrail null blocks descriptive analysis. Reconcile machine template with final prose and retain evidence links/hashes where available.

**Residual risk:** Snapshot registries remain vulnerable to overwriting or unstable IDs until repaired; published expectations remain uncalibrated.

**Status:** open at final critic review. Subsequent integration may resolve documentation without a third critic pass or empirical support.

## Top five remaining failure modes

1. Incorrect stratum-to-embryo sampling changes the target population and makes the claimed endpoint-mixture identity false.
2. Recentered endpoint variance, or output renormalization, explains an MMD gain that is incorrectly attributed to biological conditional-state improvement.
3. Unverified normalization domains, annotation semantics and mixed coordinate frames yield apparently valid arrays with invalid expression/joint estimands.
4. Assessment reuse or exposed-report retrieval remains possible because the access ledger and clean-context boundary exist only as policy illustrations.
5. Missing scorer, biological replication and pilot margins prevent confirmatory readiness; registry/template ambiguity or legacy reuse converts unresolved evidence into scientific authority.

## Final disposition

Finalize as a source-grounded audit plus a conditional, narrowed prospective recentering protocol once these corrections are reflected consistently. Keep all biological benefits untested, official scorer parity unresolved, operational access enforcement unimplemented, spatial readiness false absent frame contracts, and competition exposure unresolved. Exact practical effects, variance, power, novelty and causal interpretation cannot be filled by expert agreement. Failure to establish a benefit is not automatically falsification. No third critic loop is requested or permitted by this review.
