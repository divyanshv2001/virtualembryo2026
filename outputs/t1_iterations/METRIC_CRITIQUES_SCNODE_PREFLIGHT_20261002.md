# scNODE synthetic resource preflight critiques

Four fresh methodological specialists, role-policy version2. All benchmark metrics unavailable, reward0. Capacity checks are not biological accuracy. Proposed remedies remain unvalidated.

## DE

Problem: Finite gradients and five pretraining/three joint updates cannot establish biological convergence or DE preservation.
Proposed solution: Biological training at <=8.5 on complete mapped panel, past-derived expression anchor plus learned residual decoder. Match beta0/.1 initialization, data, batches, budget, architecture and inference sampling.
Validation: Fixed full scorer/gene mapping, held-out future for scoring only; report all four metrics and unchanged gates across paired seeds. Missing metrics, leakage, unstable gains or gate failures are inconclusive/failure.

## Direction

Problem: Synthetic feasibility says nothing about DE direction; full biological decoder is unimplemented.
Proposed solution: Past-only full mapped-gene anchor-residual decoder with latent dynamics. Beta0/.1 arms differ only regularization; select parameters exclusively using past validation.
Validation: Unchanged preprocessing, scorer and full metrics/gates. Missing scores, leakage, gate/resource violations fail. Readiness unchanged.

## MMD

Problem: No biological MMD evidence; finite gradients and low memory cannot establish distribution recovery.
Proposed solution: Past-fitted normalization/anchor/residual decoder, matched beta0/.1 arms. Preserve fixed evaluation kernel; any training choices use eligible past only.
Validation: Fixed full scorer and gene mapping/sample protocol; paired seed uncertainty without target tuning. Missing metrics, incomplete coverage, leakage or gate failure invalidate promotion.

## Variogram

Problem: Synthetic checks do not establish biological dependence recovery; no spatial coordinates exist.
Proposed solution: Specialist suggests a separate past-only gene-pair dependence penalty and matched no-penalty/generic-regularizer controls after implementing full mapped-gene anchor-residual decoding.
Validation: Unchanged scorer and four-metric gates, paired seeds, no target-informed selection; missing scores/gate failure or generic-regularizer explanation fail the hypothesis.

Coordinator: Initial paper trial remains beta0/.1 dynamic regularization only. The additional variogram-specific penalty is deferred to avoid changing multiple mechanisms and adding unmatched arms. Original promotion/readiness criteria remain; no benchmark or reward was invented for the capacity test. Detailed decoder mapping/transfer constraints must be frozen and tested before biological launch.

Private report SHA256: 216d8e03cfa49be44866f748be182450a66c3808771f1fe257566021bd2d84da
