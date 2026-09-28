# Stage conditioning and hurdle decoding

Status: completed, no promotion. No >72 gate has passed; challenge incumbent remains unit_k16.

The network adds developmental stage to the observed latent cell and noise inputs. Adjacent past-stage distributions train square-root latent MMD with 800 updates, batch64 and hidden64. All internal checkpoint selection is based on past-stage cell groups. It is not independent temporal validation. Forecasts compare freezing the last observed transition time against extrapolating time beyond the trained range. Both assumptions are explicitly tested.

Six declared variants at strength .5: freeze/legacy; extrapolate/legacy; freeze/legacy with direction gate; freeze/hurdle; freeze/hurdle with direction gate; extrapolate/hurdle with direction gate. Persistence and the unchanged state-based control accompany every fold. Historical gene directions use the latest three observed stages, suppressing reversing changes and shrinking by cell-mean uncertainty. They gate proposals before mapped mass conservation; they do not guarantee the signs of final normalized changes.

The hurdle decoder separates detection probability and positive log-abundance moments, fitting ridge regressions on past latent cells plus stage. Detection changes are capped at .01, expression factors at 1.25. New positive values use only measured source-cutoff positive abundance for genes with at least ten positive source cells. This can activate a gene absent in a particular donor; it cannot recover a gene never observed in the source. Missing/ambiguous official genes remain bit-identical, mapped abundance mass is conserved and covariance changes are guarded.

[MAST](https://link.springer.com/article/10.1186/s13059-015-0844-5) models detection and conditional positive expression as separate components. Its methods also account for cellular detection rate as a covariate. This experiment adapts the two-part idea using ridge moment decoding; it does not implement MAST logistic/Gaussian likelihoods, empirical Bayes priors or its inference procedure. Input expression is normalized abundance in log1p coordinates, so this is not a raw-count negative-binomial model. Cellular detection-rate nuisance adjustment remains untested.

The earlier quarter-day folds run before the separate one-day whole-atlas mechanism check. The broader whole-atlas cohort cannot certify the challenge-associated configuration. All evaluation retains the complete 32,285-gene four-metric scorer and invalid calibration handling. No submissions or Jev requests. Tests check numerical MMD gradients, future-expression exclusion, protected genes, mapped mass and bounded new zero activation.

## Completed quarter-day folds

All calibrations were valid. Unfiltered legacy freeze/extrapolate scored 54.2682/46.6857 and 55.4358/46.6147 across the two folds. Legacy direction filtering scored 50.4271/48.3329. Hurdle decoding without direction filtering scored 51.7722/46.7439; gated freeze/extrapolate scored 49.3260/48.2844 and 49.0610/48.2021. Persistence scored 50/50. No candidate improves both folds, so no promotion is justified.

The previous autonomous square-root run has identical floor/ceiling calibration panels, verified before comparison. The new model does not resolve its stage-generalization failure. The actual gene detection changes are recorded separately: a .01 conditional proposal cap is not a guarantee that empirical gene detection fractions change by at most .01.

## Completed one-day mechanism check

Persistence 50; state control 51.5766. Legacy freeze/extrapolate scored 48.5185/48.8290; gated legacy 50.0370; hurdle freeze 49.0035; gated hurdle freeze/extrapolate 50.2625/50.1531. All calibrations valid. Twenty-four local evaluations are complete. Zero submissions and Jev calls; no challenge evaluation for this model. No >72 gate passed.

A past-only annotation audit now exposes source coverage/proportion differences. A fixed broad tissue-proxy cohort with equal per-group caps is being prepared from the existing verified atlas. It adds neural/somitic/extraembryonic source classes and reduces erythroid dominance. Labels are coarse selection proxies, not validated challenge identities or predictor inputs. Equal source caps do not measure biological growth.
