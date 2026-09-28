# Prospective experimental design: final revision 2

No biological experiment has run. Data, independent-replicate manifests, normalization metadata and a versioned scorer are absent. The null fields in preregistration_template.json must be filled before assessment; confirmatory readiness is false.

## One primary question

T2 embryo released-stage interpolation proxy: train on E6.75/E8.0; assess once at E7.25. Compare frozen shrunk endpoint recentering versus endpoint mixture. Persistence is secondary; heart, T1, T3 and complex architectures are exploratory. This does not establish official E7.5 interpolation, extrapolation or unseen-gene performance.

D_e=MMD(endpoint_mixture,target_e)−MMD(shrunk_endpoint_recentering,target_e), larger better. Independent embryos define e; cells and seeds do not create biological n. Exact estimator/kernel/bandwidth/preprocessing/sampling must be frozen in SCORER_CONTRACT.json. Replicate/litter/batch relationships are required; unknown replication forces descriptive analysis.

## Minimal estimator

corrections/loop2_genmodel.md corrects the donor weights/means from the loop-1 recipe in one documented normalized MERFISH panel. Never pool incompatible assays. Training endpoint annotation strings define strata and unknown handling. Missing required annotations halt the stratified protocol; a single-stratum fallback is a different frozen analysis. Equal-embryo endpoint proportions give π_k=.6p_Ak+.4p_Bk. Draw k, endpoint with probabilities .6p_Ak/π_k and .4p_Bk/π_k, then embryo with probability p_esk/Σ_e p_esk and a uniform stratum row. Endpoint-mixture equals the convex equal-embryo endpoint expression mixture, not a novel forecasting method.

Use μ_sk=Σ_e p_esk μ_esk/Σ_e p_esk. Shared strata use μ_tk=.6μ_Ak+.4μ_Bk and v=X_sj+.5(μ_tk−μ_sk). Shrinkage .5 is frozen, not validated. Endpoint-only strata shift zero; unseen support has probability zero. Comparisons share donor ledgers. Keep donor coordinates only as uninterpreted placeholders unless a training-derived common frame, physical units and coverage pass separate spatial-readiness gates. Pairing does not validate pooled neighborhoods or predict growth. No target-derived registration is allowed. Spatial extensions need separate calibration/protocol.

Only documented x=log1p(Lr/Σr) permits inverse-transform/clipping/renormalization/logging. Require training round trip, reject overflow, document zero-row policy. Unknown transform halts this estimator. Both methods receive the same output map. Panel closure, denominator domain, log base, L and zero-row policy must all be documented; a wider denominator stops this map. Annotation method/version, reserved-token collision refusal and allowed missing-label mass must be frozen. Incompatible provenance stops this stratified analysis. Before the map whole-row sampling preserves donor residuals; afterward covariance changes. Stream B=256 rows with O(Kd+Bd) memory and float64 sums, record PCG64 seeds/checksums. Avoid dense 32,285-gene covariance. baseline_contract_results.json contains arithmetic illustrations, not a biological implementation or scorer.

## Access boundary

Training permits fitting; separately eligible development data permit selection. Freeze estimator/hyperparameters before assessment. Record every target access in an append-only ledger; assessment irreversibly consumes its confirmatory role. Residual-guided model changes are exploratory and require a fresh eligible assessment. Three snapshots offer no automatic nested stage split. Legacy _proxy_score is prohibited as evidence for this design.

## Decision and uncertainty

Freeze δ*>0, μ_A>δ*, α, power and paired variance σ² from eligible pilot evidence/scientific tolerances before assessment. H0:E[D]≤δ*. Practical benefit requires a one-sided lower confidence bound >δ*. Planning approximation: n≈[(z_(1−α)+z_(1−β))σ/(μ_A−δ*)]². Halving headroom quadruples n, verified by the arithmetic checks. Small-sample and clustering analyses must match the manifests. Insufficient independent n permits descriptive D and seed sensitivity only, without population intervals, power or confirmed benefit.

Secondary official components are descriptive. **No no-degradation claim is made.** A later joint claim requires deterioration-oriented G_j, tolerances η_j and simultaneous upper bounds below η_j. Nonsignificant harm is insufficient. Numerical margins remain null. Never compare local proxy scores with official anchors or leaderboard baselines.

## Controls and interpretation

Future checks: whole-cloud/first-row sampling, panel/order/normalization contracts, unsupported strata, expression-position permutation, units/rigid-frame/scale sensitivity, subset-neighborhood reconstruction, graph-free/randomized-prior controls if graphs are added, equal selection budgets. Expected invariances require exact scorer contracts. Synthetic sensitivity is distinct from anatomical validation.

Failure to clear δ* fails to establish practical benefit; strong measured negative effects may contradict it. Insufficient evidence remains untested/ambiguous, not falsified. Independent imaging and full transcriptomes test anatomy/state; lineage tracing tests ancestry; restricted interventions, occupancy and rescue test mechanism. Chronological and maturity-adjusted comparisons answer different estimands.

EXPOSURE_REGISTER.md records irreversible protected-target literature exposure. This audit cannot certify competition eligibility. Future competition design needs a separate clean context and screened provenance; target-informed audit recommendations cannot choose competition priors/tissues/hyperparameters. Audit registries exclusively govern scientific statuses; legacy ACT/LEARN/rank and Jev do not establish biological evidence.

## Loop-2 controlling boundaries

SYNTHESIS_2.md controls final interpretation. The mixture already changes conditional expression distributions; recentering tests a frozen transformation, not identified composition/state biology. With equal proportions it can leave mean unchanged and quarter between-endpoint variance before output mapping.

The access flag/event sequence is a policy illustration only. Durable enforcement and clean-context isolation are unimplemented and untested. Real launch must reject repeat assessment, changed locks and excluded reads under immutable hashes. Expression descriptive/confirmatory and spatial readiness are separate predicates in the final template. Secondary metrics remain descriptive; guardrails are conditional/not applicable.
