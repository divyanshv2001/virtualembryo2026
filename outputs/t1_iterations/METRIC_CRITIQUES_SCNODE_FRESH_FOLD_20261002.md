# Fresh scNODE source-domain temporal test — four specialists

Private report SHA256: 5225e5a3de644e754b5089dc91662ebd4ce2ea7ab6d351e0b08ac0de8bda6872. Specialist role-policy version2. Four fresh agents, each <=150words. Source8.5 previously exposed; three resamples are not independent embryos. Raw4/skills4 retained unchanged in report and ledger.

## de_score

**Problem:** Both variants have negative raw DE and mean skills beta0 .45677/beta.1 .45552 below freshCNF .46186 and copy.5. Calibration validity does not establish DE fidelity; uncertainty limited by three exposed resamples.

**Proposed solution:** Add training-only DE consistency to the existing anchored residual, penalizing direction/rank errors on transitions before8.0. Select weight using earlier held-out transitions then freeze.

**Validation:** Compare unchanged residual, CNF and copy with matched seeds/batches/noise/Euler. Require positive rawDE and reproducible paired DE gains over controls without headline degradation; otherwise reject. Provisional comparison, not certification.

## de_direction

**Problem:** Mean skills beta0 .49537/beta.1 .48531 trail copy.5 andCNF .50856; all rawdirections negative. Final driftReLU may restrict signed motion, causality unproven. Dynamic regularization worsened this fold.

**Proposed solution:** Earlier past-only fold ablates only finaldrift activation ReLU/linear/tanh initially beta0, matched initialization/budget/draws/anchoring/.0625Euler. Inspect latent signs and decoded change alignment.

**Validation:** Freeze before future reads; require positive raw direction all prespecified resamples and mean above controls. Failed improvement rejects activation hypothesis; no official gain/readiness.

## mmd_u

**Problem:** Beta0MMD .51137 narrowly exceedsCNF .50886;beta.1 .50120 trails it. Both headlinesbelowcopy. Removals145883/238922 vsadditions18343/64900 suggest sparsification, not establishedcause. Old contraction measurements do not describe freshmodels.

**Proposed solution:** Fit past-only gene detection thresholds constraining deletion/addition balance to past-transition ranges; test uncorrected and constrained correction with fixed massguard.

**Validation:** Hold traininghashes/fold/seeds/Euler/scorer/calibration fixed, compare allcontrols. Require reproducible MMD and headline gains; reject better detection balance without predictive gains.

## variogram

**Problem:** Beta0/beta.1 .49349/.46684 trailCNF .49907/copy.5. Gene-pair benchmark requires no coordinates. Mass conservation does not preserve gene-pair structure; thinning/decoder hypothesis remains uncertain.

**Proposed solution:** Past-only earlier pseudofolds attribute error to detection, marginal shifts and residualcovariance. If detection dominates, test past-estimated detection/covariance penalties in existing residual.

**Validation:** Matched copy/CNF/betas/remedy, seeds/inputs/mass/.0625Euler. Require consistently lower pastfold error without worse detection calibration. Reject inconsistency or persistentcovariance distortion; no exposedfuturescore tuning.

## Coordinator decision

Reject both newarms; no16-resample expansion or submission. Copy50, freshCNF49.50783, beta0 49.01448, beta.1 47.89373. Original gates unchanged. Separate predeclared reward vsfreshCNF: beta0 -29 (MMD+1, other3 -10), beta.1 -40, batch -69, balance -1124. Controls0; requested additional30-point penalty pending user evaluation remains separate.

Jev synthesis chose past_decoder_detection_attribution confidence.98, one1942byte summary call, observed956input/65outputtokens. Before adding losses or switching papers, freeze training-fit component audit of these fresh checkpoints using only<=8.0 expression. Explicit in-sample7.75to8.0 reconstruction is not a fresh hindcast. Guard backoffs may differ between components; report this confound. No benchmark scores/proxyreward. Use findings to predeclare one fresh earlier-cutoff remedy.
