# Critic loop 2 — donor weighting and interpretation correction

I independently verified the critic's correction by deriving the joint sampling law. Loop 1's uniform choice among embryos containing a stratum was internally compatible with its own conditional mean, but incompatible with the stated equal-embryo endpoint population distribution. I accept the error. This document supersedes that donor/mean rule and narrows the interpretation; loop 1 remains unchanged as history. No biological data or performance results are available.

## Corrected population and conditional estimator

For endpoint s, E_s training embryos, embryo e with n_es rows, stratum k has n_esk rows and p_esk=n_esk/n_es. The intended endpoint population first samples e uniformly from all E_s embryos, then samples a row uniformly from that embryo. Hence

p_sk = (1/E_s) sum_e p_esk,

P(e | s,k) = p_esk / sum_f p_fsk.

Embryos with p_esk=0 have zero conditional weight, but remain in the endpoint proportion denominator E_s. For every positive-count embryo/stratum, mu_esk is its within-stratum row mean. The correct endpoint conditional mean is

mu_sk = [sum_e p_esk mu_esk] / [sum_e p_esk].

Never evaluate mu_esk for an absent stratum; its numerator contribution is zero. If the denominator is zero, that endpoint has no conditional distribution for k. The previous unweighted mean of positive-count embryo means is withdrawn.

Retain A=E6.75, B=E8.0, target proxy E7.25, w=0.4, and t_A=0.6, t_B=0.4. Set pi_k=sum_s t_s p_sk. Sample k with pi_k, then s with q_sk=t_s p_sk/pi_k, then e with the corrected conditional weights, then a whole donor row uniformly from the n_esk stratum rows. This joint law reduces, for any donor row belonging to s,e,k, to

P(s,e,j)=t_s / (E_s n_es).

Indeed pi_k q_sk [p_esk/sum_f p_fsk] / n_esk equals that quantity. Thus the mixture baseline is exactly a convex mixture of two equal-embryo endpoint empirical distributions. The derivation provides an implementation contract independent of any target values.

For shared strata retain mu_tk=0.6 mu_Ak+0.4 mu_Bk and fixed lambda=1/2. Raw mixture output is X_sj. Raw recentered output is X_sj+lambda(mu_tk−mu_sk). Their conditional means are m_mix,k=sum_s q_sk mu_sk and m_rec,k=(1−lambda)m_mix,k+lambda mu_tk. Endpoint-only strata still receive zero shift. Apply the same verified assay-specific output map to both; after clipping/renormalization these mean formulas need not remain exact. Training-only strata, UNKNOWN handling, whole-row coordinate pairing, PCG64 donor ledger, normalization gates and memory limits from loop 1 otherwise stand.

## Small worked example — algebra only

Take two training embryos at A with 10 rows each. Stratum k occupies 8 and 2 rows, with expression means 1 and 5. Then p_Ak=(0.8+0.2)/2=0.5, conditional embryo weights are 0.8 and 0.2, and mu_Ak=0.8×1+0.2×5=1.8. Uniform positive-embryo selection would incorrectly produce mean 3 for this intended population.

At B take two 10-row embryos with 4 and 1 k rows, and means 3 and 8. Then p_Bk=0.25, conditional weights again are 0.8 and 0.2, and mu_Bk=4. The proxy mixture weight is pi_k=0.6×0.5+0.4×0.25=0.4. Conditional endpoint weights are q_Ak=0.75 and q_Bk=0.25. Consequently m_mix,k=2.35, mu_tk=2.68, and m_rec,k=2.515. These are raw-space expectation calculations, not model performance or normalized biological predictions.

An A donor row has unconditional probability 0.6/(2×10)=0.03 regardless of embryo. An A/k row under the incorrect rule instead has probability 0.15/8=0.01875 in embryo 1 and 0.15/2=0.075 in embryo 2. The error changes the intended population and cannot be repaired by paired seeds alone.

## Narrower interpretation and verification

The term “composition-only” overstates the contrast. The empirical endpoint mixture already changes within-stratum distributions through endpoint reweighting. The actual comparison is **endpoint mixture versus that same mixture with shrunk stratum recentering**. It holds sampled endpoint/embryo/row identities and pi_k fixed, not the entire conditional expression distribution. A gain would concern this predictive recentering intervention, not identified biological decomposition into composition and state.

Before the nonlinear output map, within-endpoint/stratum covariance is unchanged by translation. Across endpoints within k, the between-endpoint covariance contribution scales by (1−lambda)^2, because the endpoint mean difference becomes (1−lambda)(mu_Ak−mu_Bk). For this example its scalar contribution falls from 0.75×0.25×2.2²=0.9075 to 0.226875; within-endpoint variance is additional and unchanged. Thus even a raw-space gain can reflect variance contraction as well as mean recentering. Clipping and normalization add further distributional effects. No full-covariance estimation is needed to implement this model.

Future tiny contract checks should assert the joint probability identity, conditional means and zero-denominator handling exactly, including unequal embryo capture sizes and unequal stratum fractions. Whole-row/coordinate pairing should be checked against the corrected ledger. Synthetic checks establish algebra and implementation, not biological validity, empirical uncertainty, normalization metadata or scorer parity. Final model selection and assessment remain separated, with target values unavailable to fitting or selection.
