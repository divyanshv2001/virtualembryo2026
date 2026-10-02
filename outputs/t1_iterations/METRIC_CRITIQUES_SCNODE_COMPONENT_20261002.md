# Fresh scNODE component audit — four specialists

Private reportSHA71500248bb2b6248b9310f2cae50472b1eb8f0bf6c68494b82b714e5ef8448d0; specialist role-policyversion2. Eachfreshreview<=150words. Training-fit7.75to8.0, weights/heads already saw8.0. No benchmark rawmetrics/skills or reward. Protectedgenes/mappedmass passed. Component backoffs differ; no causal attribution or accuracy claim.

## de_score

**Problem:** DE unavailable. Decodedvariance.09551/.10759, signagreement.08195/.11060 over18291activepastgenes includingpredzero;cosine-.00715/.38628. Zero negative latentvelocity suggests restriction, notproof.

**Proposed solution:** Matchedbeta.1 signedlinearfinaldrift vsunchanged withsame anchoredresidual/architecture/budget/seeds/caps, earlierfit7.5forecast7.75.

**Validation:** Compareunchanged/persistence atsamecutoff; direction/signagreementwithoutworsemeanchangeerror/guardviolations. DiagnosticsarenotDEscore; rejectfailedhypothesis.

## de_direction

**Problem:** Endpoint8.0fit makesalignmentdescriptive. Joint.0188/.4166 vsdecoded-.007/.386;beta0jointbackoff.5vscomponent1 confounds. ReLU forbidsnegative latentvelocity butsigned decodedmotionpossible. Decoder retains.0955/.1076variance.

**Proposed solution:** Fresh<=7.75fit→8.0, signed/ReLU acrossbetas0/.1 withmatchedinit/budget/donors/sampling/strengths/backoffs,include persistence.

**Validation:** Prespecifydirection/variance/reconstruction/conservation; improvecontrolswithoutcollapseorviolations,otherwisereject. No scores/reward/readinessestablished.

## mmd_u

**Problem:** MMD unavailable. Variance.09551/.10759 anddecodedDetection.13030/.13072 vsobs.12096 suggestcompression/miscalibration, notMMD. Endpointfit/unequalbackoffspreventcausality.

**Proposed solution:** Fresh<=7.75 stochastic hurdledecoder withpastcalibrateddetection, conditionalpositive dispersion andcorrelatedreconstructionresiduals; unchanged/detectiononly/dispersiononly/botharms hold drift/beta/data/init/budget/draws/backofffixed.

**Validation:** Earlierrolling-originselection, frozenfullscorer/kernel; lowerMMDwithpaireduncertainty, betterdetection/covariance andguards. Failureuncertaintyincludesnogain,covarianceworse orconstraintviolations. Reviewer called8.0untouched; correctedbelow.

## variogram

**Problem:** Gene-pairstatistic unavailable;variance/deletionnumbers areproxies and8.0inclusivefit cannotestablishperformance/cause.

**Proposed solution:** Fresh<=7.75 bothbetas, deterministicvsdecoderdistribution calibratedpastonly, reference/componentcontrolssamecells/genes/seeds/sampling/guards. Beta0jointbackoff.5others1.

**Validation:** Unchangedscorer, lowerheldout rawvariogram/higherskillvsreferencewithguards; missingscores/nogain/failure rejects. Reviewer called8.0untouched; correctedbelow.

## Coordinator corrections and decision

E8.0 is already exposed in training/audit: a fresh cutoff-fit test is retrospective, not untouched. Variogram needs no spatial coordinates. DE suggested fitting only7.5: one snapshot cannot train temporalvelocity; retain7.5/7.75 history. Existing observed-donor residual anchoring means mean-decoder variance compression is not proof that finalforecasts lose the samevariance. Do not add fourdecoderchanges now or expandbetagrid.

Jev chose one isolated signed-drift beta.1 trial, confidence.84;2178byte summary, observed1081input/69outputtokens, single request. Fresh7.75cutoff, same200pretrain+1000jointsteps forReLU/linear, identicalcoefficients/objective/decoder/draws/.0625Euler, copy/freshCNFcontrols. Fixedguardbackoffsmaymediateeffect, recordthem withouttuning. Full32285scorer/calibrationunchanged. No16rep expansion unlessallheadline/meanskillcontrols beatenandmean>=60; original>72/lower-tail64rep/wholeembryogatesunchanged. Iftrialfailscloseactivationremedy, noautomaticactivationgrid. Thisaudit changes no score/reward.
