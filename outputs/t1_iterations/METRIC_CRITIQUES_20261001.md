# Four metric critiques

Four agents reviewed targeted frozen evidence on the user's explicit request.
These are scientific critiques, not measured new experiments. Full evidence:
CNF_COVARIANCE_ALIGNMENT_RESULTS.json, EB_GENE_SLOPE_SCREEN_RESULTS.json and
the formula_progress_20260930_01 submission report.

| Metric | Finding versus mean-shift incumbent | Next falsifiable mechanism |
| --- | --- | --- |
| DES / de_score | Covariance .25 leaves DES unchanged on all three panels, mean52.563. EB proxy fails. | Lineage-conditioned temporal residual drift separating abundance changes from within-lineage gene changes. |
| DCS / de_direction | Mean60.087 to60.217, but third panel falls59.292 to59.181. All four EB variants worsen direction on both folds. | Past-state-adjusted positive-expression/detection residual slopes; freeze alpha .1/.25 before held-out reads. |
| MMD / mmd_u | Mean skill56.24 to56.68, lower raw MMD on all panels; .5 trades away DE. | Conditional residual drift targeting non-Gaussian distribution mismatch, matched donors and incumbent. |
| CSS / variogram | Mean53.596 to53.947, all three panels improve. | Within-lineage joint detection coupling, strengths0/.25/.5 at fixed gene detection counts and positive-value margins. |

DES critic: critique_de. DCS critic: critique_direction. MMD critic:
critique_mmd. CSS critic: critique_variogram. All four reviews completed.

Scorer CSS compares mean square-root absolute gene-pair differences; latent
covariance alone does not determine it. Raw MMD is lower-is-better; normalized
skills are higher-is-better. A positive mean does not hide panel regressions.

Freeze lineage definitions, residual estimator, correction strengths, controls,
splits and acceptance before a target read. First require gains across historical
cutoffs; then evaluate unchanged full-panel metrics. Preserve mass/protected-gene
guards, and record failures rather than quietly changing them. Sparse associations,
composition confounding and mass correction can erase proposed coupling benefits.

Cell resampling is not independent embryo or temporal validation. Reused E9.5
panels invite selection bias. Proxy overlap cannot certify DES or E10.5 transfer.
Submission format checks establish format only. The export uses source through
E9.5 and17057 E9.5 anchors; generic E8.5 audit wording is annotated as stale.

The new +1/-10 research reward starts at0 and has a100 maximum. Future valid,
predeclared paired mean gains earn+1 per metric; ties/regressions earn-10.
Invalid, pending or proxy evidence earns0. Negative balances remain visible.
No retroactive award for these three reused panels. Actual benchmark scores and
the >72 temporal/64-replicate mean/lower-tail gates remain unchanged.

## Real full-gene decoder — 5 October 2026 IST
Report hash: cb96904c91a185a00d8a57716cda8d0aa48904ca597393de01652ba3a71477df; four fresh narrow reviews; primary nonlinear/linear reward-40.

- de_score: Problem: NeuralDE trailslinear/incumbent; exposeddevelopment notindependent. Proposed solution: Preserve linearpartialgains; inspectdetectionmismatch beforetraining. Validation: Frozencontrols/freshpasttransfer, no weightgrids.

- de_direction: Problem: NeuralDCS .556592 trailslinear .609334/incumbent .602168. Proposed solution: Preserve linearDE/DCS/MMDgains, diagnose zerofilling/covariateshift. Validation: Matchedpastforecasts/allfourmetrics, CSS mustrecover.

- mmd_u: Problem: NeuralMMD regresses; linearMMDpartialgain retained. Proposed solution: Diagnose detectiondensity/librarynormalization/covariance beforetraining. Validation: Pairedfullpanel outcomes acrossallfourmetrics; no oldzero-lock/mass grids.

- variogram: Problem: NeuralCSS .126780 vslinear .370882/incumbent .539468. Proposed solution: Quantify newlypositiveentries/clipping/libraryrescaling beforemodelchanges. Validation: Exactforecastreplay withfixedsettings; no hiddenfuture. Coordinator: reviewer lowermeanerror claim was unsupported and excluded; newzeros means newlypositive donorzeros.

Exact replay subsequently measured10633/12303newpositivegenes/cell, preserving observed donor reference. CachedJev .11/1089in54out chooses hurdle novelty preflight; no repeated zero-lock/mass grid authorized. Partial linearDE/DCS/MMDgains retained; no readiness or officialscore claim.

## Observed full-gene hurdle — 5 October 2026 IST
Report 03c1b7ec9ef8808ea3bb53e805e3b5f0c9a5e8aa1545302cd75a4a3c8947d976; four fresh narrow specialists.

- de_score: Problem: Small DEgain and dependentresamples. Proposed solution: Freeze hurdle, confirm permissible temporal/training support. Validation: Matchedcopy/incumbent, no retuning orofficialclaim.

- de_direction: Problem: DCSbeatsinc buttrailsdense. Proposed solution: FreezeCNF/.25map,confirm newpast/trainingsupport. Validation:3exposedresamples/12validcalibs do notestablishindependence.

- mmd_u: Problem: MMDimprovesoverinc buttrailsdense. Proposed solution: Freeze gains; confirm matcheduncertainty/temporal support. Validation:3exposedresamples/12calibrations are notindependentembryos.

- variogram: Problem: Promotion lacksconfirmed temporal/training support. Proposed solution: Freeze decoder/covariance;confirm support. Validation: CSS.555417>.539468,allfourmeansimprove,12validcalibs/guardpass. Coordinator: reviewer full-grid meansCSS, no gridrun.

## Fixed hurdle head stability — 5 October 2026 IST
Report 1e053dc8a8375ed99f4aaeb80131401c46882cbb06040cc8377a22029f42b0fa; four fresh narrow reviews.

- de_score: Problem: Half0DCSregression failsdeclaredstability. Proposed solution: Retainfrozenfullfit;no posthochalf1selection/independentembryoclaim. Validation: Exactreplay/DEgains do notestablishstability.

- de_direction: Problem: Half0DCS.601513<inc.602168. Proposed solution: Retainpilot withoutstability/promotionclaims;rejecthalf1posthocselection. Validation: Bothhalves mustpass; biologicalreplication remainsunverified.

- mmd_u: Problem: DespiteMMD/aggregate gains,half0DCS failsboth-halves/allfourcriterion. Proposed solution: Preservepilot,rejectpromotion/posthochalfselection. Validation:15validcalibs/exactreplays notindependentembryo/temporalrobustness.

- variogram: Problem: Half0DCSfailsdeclaredstability. Proposed solution: Preservefullfitprovisionally,seekgenuine temporal/embryo support;no grids. Validation: Exactreplays/3exposedresamples cannotestablishgeneralization.

## Fixed ridge2 hurdle stabilization — 5 October 2026 IST
Report 738da7c337aabaaf3aaecc3805202de009a9383057d6cfe4f9c36e7d693f9da6; four fresh narrow read-only reviews, factual comparator corrections verified.

- de_score: Problem: Higher de_score improved, but the promotion gate failed and MMD regressed; no independent embryos support generalization. Proposed solution: Preserve gains without promotion or posthoc arm selection. Refit ridge strength using past-only nested validation. Validation: Freeze incumbent controls; require both halves/all four checks and full mean >= original. Reject MMD regression.

- de_direction: Problem: Ridge2 improves direction (.60616 versus .60494) but fails stability gates; exposed-panel repair and no independent embryos prevent promotion. Proposed solution: Freeze ridge2; develop remedies using past-only data with matched gene, split, and MMD controls. Validation: Require both halves/all four and full mean>=original on untouched embryos; otherwise reject promotion. Quantify uncertainty; valid calibrations alone cannot establish stability.

- mmd_u: Problem: Aggregate improves versus incumbent; MMDmean regresses versus incumbent and originalfull, improving versus copy only. The gate fails. Proposed solution: Preserve partial gains without promotion or posthoc selection. Keep calibration pinned; use past-only tuning with frozen incumbent, originalfull, and copy controls. Validation: Require both halves/all four and fullmean>=original. Any failure blocks promotion; embryo independence remains unverified without IDs.

- variogram: Problem: Ridge2 variogram mean improves versus incumbent (.554715 versus .539468), trails original (.555417); aggregate likewise trails (56.498 versus 56.689). MMD mean regresses versus incumbent, improves versus copy only. Proposed solution: Preserve variogram gains; use past-only tuning, matched controls, locked failure criteria; no promotion. Validation: Keep calibration unchanged; require both-halves/all-four and full-mean gates, quantify uncertainty. Independent embryos unavailable.

Coordinator: nested tuning and independent embryo suggestions remain unvalidated/unavailable; no score-informed grid or calibration change. All18 calibrations valid; both halves repair mean direction versus incumbent but allthree newarms regress MMD. Primary fullfit versus originalfullfit +1/+1/-10/-10=-18; half-fit diagnostics earn no separate reward.

## Frozen detection channel comparison — 5 October 2026 IST
Report c0212a2b082c7cea8caed5634338e68708a0e71fd70317d1ef8e8ef538736215; four fresh narrow read-only reviews, factual comparator corrections verified.

- de_score: Problem: DE improves, but MMD/variogram remain slightly below old-full; no independent embryos establish temporal readiness. Proposed solution: Export the frozen candidate for prospective submission only, retaining controls and failure criteria; do not upload. Validation: All18 calibrations and retention gates pass. Fixed E8.5 cell halves support robustness, while exposed E9.5 targets limit independent validation.

- de_direction: Problem: Mean-based gate passes, but half0 seed3 regresses; independent embryo and temporal stability remain untested. Proposed solution: Export prospectively with frozen settings, retain the original failure, and add independent embryo/temporal controls plus prespecified per-seed regression limits. Validation: Confirm gains across both halves and aggregate; fail readiness if independent controls or any prespecified regression limit fails.

- mmd_u: Problem: MMD improves over incumbent but slightly regresses versus old full; exposed E9.5 and fixed E8.5 halves cannot establish independent readiness. Proposed solution: Export only; validate on untouched past embryos with controls frozen. Validation: Current retention gates pass across18 valid calibrations. Fail readiness if independent embryos are unavailable or prespecified gates fail; no causal certification.

- variogram: Problem: Full variogram skill slightly trails old-full despite aggregate improvement; embryo independence and temporal readiness remain unproven. Proposed solution: Export frozen positive1/detection2 controls with past-only calibration; reject updates failing either-half/all-four improvement versus incumbent or full-retention gates. Validation: Current gates pass across18 valid calibrations. Exposed E9.5/E8.5 halves limit generalization; no biological causal claim.

Coordinator: Independent embryos remain unavailable; per-seed criteria are suggestions, not changes to the current gate. All18 calibrations valid; fullfit/bothhalves pass declareddevelopmentgate. Slight fullMMD/CSSloss vsoriginal retained; no officialreadinessclaim. Primary fullfit versus originalfullfit +1/+1/-10/-10=-18; half-fit diagnostics earn no separate reward.
