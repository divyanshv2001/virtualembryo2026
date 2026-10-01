# Capture OLS gate decoder specialist reviews

Private report SHA256 `8d2dbba3ed81cbca4abc748a7a11de89d9ad8b146382f9f8705519d3a7736afd`; role-policy version2. Four reused existing specialist contexts, observed thread-limit limitation disclosed. No futureexpression/scorer/reward.

## DE

Problem: DE metrics are unavailable. All four methods fail the early mean guard, although the curvature gate passes covariance. Later guard success does not establish accuracy. Millions of clipped entries indicate that decoding substantially changes the proposed shifts; omission sign agreement alone cannot preserve their intended DE behavior.

Proposed solution: perform a bounded, past-only decoder audit using the existing forecasts. For each lineage, measure gene-sign reversals, newly created zeros, and changes in gene ranking and shift energy before versus after clipping and mass restoration. Attribute mean-guard error to those operations. This tests whether failure originates in slope estimation or decoding before introducing another method.

Validation: replay copy through the same audit, retain unsupported groups, and compare all four fixed candidates on identical genes and coverage. If clipping drives distortion, test a positivity-preserving multiplicative decoder with parameters fixed through past validation. Keep guards unchanged and freeze forecasts before future scoring. Readiness remains unchanged.

## Direction

**Problem:** DE direction remains unavailable. All four methods fail the early mean guard; curvature passes covariance but still fails mean, so sign screening alone does not control decoded perturbation. Large clipping counts suggest decoder distortion, but counts do not establish sign reversals or their cause. Passing later guards establishes feasibility, not accuracy.

**Proposed solution:** First audit each gene’s requested versus realized mean shift after decoding, clipping, and mass repair, separately by lineage. Attribute sign reversals and excess perturbations to each step. If clipping/repair causes reversals, predeclare a constrained decoder that preserves requested gene means subject to nonnegativity, protected genes, and existing mass constraints; infeasible requests must be reported and rejected.

**Validation:** Replay identical frozen shifts through both decoders, with copy and incumbent controls. Require existing guards plus preserved requested signs before scoring; subsequently require direction gains on both folds. Shared-middle interval agreement is not causal evidence. Captures are not embryos; critic context is reused.

## MMD

**Problem:** MMD is unavailable. All four decoders fail the early mean guard; three also fail covariance. Later feasibility does not establish forecasting accuracy. Large clipping counts suggest a decoder mismatch, but counts alone cannot identify its cause. Clipping and mass repair can change covariance despite translation preserving it beforehand.

**Proposed solution:** Run a past-only, staged transformation audit on the frozen shifts: record distributions before translation, after translation, after clipping, and after mass repair. Decompose mean and covariance changes by lineage, gene detection frequency, and donor expression quantile. Test whether early violations concentrate in donor-zero genes receiving negative shifts or arise mainly during mass redistribution. Keep methods and guard thresholds fixed.

**Validation:** Use identical donors and coverage for all four methods, with copy and incumbent controls. Attribute failure only when a specific transformation stage explains the guard crossing across supported captures. Otherwise retain decoder incompatibility as unresolved. No promotion or distribution-accuracy claim follows from this audit.

## Cell-state structure

**Problem:** All four decoders fail the early mean guard; three also fail covariance. Later passes show cutoff dependence, not CSS accuracy. Reduced clipping under sign gates does not establish benefit. The covariance guard covers encoder features, leaving full-panel cell-state structure unverified; CSS metrics remain unavailable.

**Proposed solution:** Run a fixed past-only boundary-localization audit. Within each lineage, partition cells by observed zero burden and intended updates by distance to the nonnegative boundary. Measure mean drift and pairwise gene-contrast distortion before clipping, after clipping, and after mass repair. This distinguishes trajectory errors from boundary effects and repair propagation without adjusting strengths or guards.

**Validation:** Use identical cells and coverage for all four methods plus anchor; include unsupported cases. A decoder-induced explanation requires distortion to arise after clipping/repair and concentrate in boundary strata. Pre-clipping distortion points upstream. Neither result permits promotion. Captures are not embryos; existing critic context is reused because of the thread limit.

## Coordinator interpretation

All4newmethodsfailcut8 meanchange(.742–.784>.5), OLS/omission/rank2 additionallycovarianceguard(.425/.425/.410>.4);curvaturecovariance.382passes. All4passcut8.25(.375–.422mean/.293–.337covariance). Masserror<=6.4e-8,finite/protected/fallbackallpass;copy/incumbenthashesexact. Curvaturegateclippingdecreases butaccuracyunmeasured. Mandatoryomissionunsupported numericalnegativecontrol verified withoutchangingexecutedsource.

Millionsofclippedentries showboundaryinteraction, butnotalone thecause ofguardviolation orgenesignreversal. Prior exactlogmean+countmass feasibility/solver failures retained: directionsuggestion isnotauthorization torepeat infeasible solver withoutnewfeasibilityevidence. Constrainedpreservedmeans maynotexist. Preclipnegative values areinvalidforecast andwillonlybe diagnostic, neverscored. Puretranslationwithinlineagepreservescovariance, butpooledlineagecentroid shifts canchangepooledcovariance evenbeforeclip. FullpanelCSS notestablishedbyencoderfeatureguard.

Jev1028input/59outputtokens,2,126-bytepacket chose staged_mean_boundary_audit (confidence1.0 advisoryonly). Fixedbeforeclip/afterclip/aftercountmassrepair means/signs/covariance andtopguardgene/pastcapturemeans, boundarystrata, all12hashreplay. No targetrank/strengthgrid orguardretuning. Specseparatelydeclared, notyetlaunched; −848/ledger/readinessunchanged.
