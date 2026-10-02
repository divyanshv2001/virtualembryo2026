# Soft OT first paper-objective pilot critiques

Private report SHA b1fc07d9c1eb4d8d1ac2ec4dcd59a8fad55a6b9f45c7db88c7dbb217ce5f993f. Fresh metric specialists role2, <=150words each; no claimed human credentials. All12calibrationsvalid, exactcopy/incumbentreplay andoldfloor/ceiling checks. No passingcandidate ornewartifact; officialbest52.16unchanged.

The overall localmean headlines are softOT40.96462, incumbent55.98883, matchedNLLcontinuation55.08680, copy50. These are not individual DE/direction/MMD/variogram values. Some review wording conflated headlines with the specialist metric; the actualraw/skills in the report remain authoritative. Adaptation lacks fullscFMbidirectional/globalobserveddistribution objectives and doesnotrefutepaper. Trainingstreams differ (OT20261002/NLL20260928) despite matchedinit/400steps, so causal attribution is limited.

## DE — soft_pilot_de

Problem: Allthree DE evaluations regress despitevalidcalibration. Stochasticcouplings oralignment mayattenuatecontrasts, butcause unidentified. Proposed solution: retainNLLanchor andtestfixedpastcouplingbank averagedacrossbootstrapdraws; chooseOTweight usingeligibleearliervalidation. Validation: same800stepinit,400stepbudgets, fixedPCA/decoder/cov.25; compareNLL, currentOT, fixedbankOT, optionalalignmentablation. RequireDEgainovermatchedNLL withoutothermetricregression; otherwise reject. Reused9.5panels exploratory, no futuretruth orreadinessclaim. Coordinator: bootstrapbank/alignmentchanges withheld fromnextsingleanchoringablation; earlierfoldencoder/init mustfitthroughthatfoldcutoff.

## Direction — soft_pilot_direction

Problem: Direction deteriorates inallthreepanels; validcalibration doesnotmakecurrentadaptation predictive. No fullscFM global/bidirectional formulation. Proposed solution: boundedpast-onlyOTauxiliaryloss retainingNLL, cappedweight/gradientcontribution. Validation: same800stepinit and400steps; explicitlymatch trainingseed/streams and compareNLL-only/boundedOT/original controls. Requiredirectionimprovement withoutotherthreeregressions; otherwise reject. Original72/64replicate/temporal readiness unchanged.

## MMD — soft_pilot_mmd

Problem: MeanrawMMD.052599 worse thanincumbent.022641,NLL.023452,copy.028091; allthreepanelsregress. Differenttrainingseeds leavecausaluncertainty. Proposed solution: fixedPCA/decoder/cov.25/sharedinit, lowweightOT withlikelihoodanchor andcouplingdiagnostics before newVAE. Validation: matchedseed/budget/draws, onlypasteligibleselection; consistentlylowerMMD vsbothcontrols andno companionregression. Otherwise reject. No E10.5truth orpaperrefutation.

## Variogram — soft_pilot_variogram

Problem: SoftOTraw.002005-.002384 versusincumbent.000934-.001033,NLL.000983-.001079; nonspatial covariance errorworse onallthreepanels. Proposed solution: retaincov.25/sharedinit; reducedOTweight, NLLanchor andparameterdisplacementcap beforechangingVAE. Validation: same400stepbudget/init,batches/evaluation; explicitmatchedrandomstreams, variogramgain withoutothermetricregression. Rejectotherwise. Thislimitedadaptation doesnotestablishfullscFMperformance.

## Decision

Allfourpairedmeanskill gainsnegative; separatefuture-onlyreward-40, controls0. Extra usermanual-30 remains pendingevaluation, notapplied. The firstpilot is scientifically rejected; no export.

Jev distinctsynthesis2430bytes1183input/65output selectedone bounded anchoredsoftOTremedy (confidence.95), before switching representation. No retries/countdrivencalls orunmeasuredsavingsclaims. Atmostone suchremedy inthissequence; then revisit scNODEdynamicregularization ratherthan endlessunanchoredvariants. NEXT_ANCHORED_SOFT_OT_PILOT.json ispredeclared butnotimplemented/launched. Data/eligibilityscope andoriginalgates unchanged.

Ledgerindex repair: finalpilot exposesidenticalpanels/folds; canonicalfolds indexedonce, distinctpanelsretained. Historicaldiagnostic scorecopies explicitlylabelledhistorical_score_replay. Regressionchecks verifiedidenticaldedup/distinctpreservation andoriginalscorermanifest match. SCORE_LEDGER:3142records/207reports/0errors, includinghistoricalreplays/controls; not3142 independentexperiments.
