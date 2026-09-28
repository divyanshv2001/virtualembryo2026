# Metric research paths

The score remains below 72. This is a tracked research queue, not a claim that literature guarantees leaderboard gains. Every experiment retains all four official-panel metrics. Methods fitted to earlier stages may be evaluated on later stages; later-stage expression never enters the learner.

The machine-readable queue records source-reading coverage, implementation, full results and unresolved paths. Abstract-only reviews require full method review before claiming faithful reproduction. Research families remain open even where a bounded local ablation failed.

## de_reliability

Regularize unreliable gene effects learned from two historical forecasts. This ridge gain implementation is an adaptation, not DESeq2 or adaptive-shrinkage reproduction.

Status: implemented_evaluated. Six blends .25/.5/1 for expression or expression+detection. Best mean 55.3744; gain .0621; one panel regresses.

Sources: [DESeq2](https://pmc.ncbi.nlm.nih.gov/articles/PMC4302049/) (abstract and selected methods), [Adaptive shrinkage](https://pmc.ncbi.nlm.nih.gov/articles/PMC5379932/) (article and sign-error method discussion)

Next: Test sign-aware empirical Bayes estimator and historical fold transfer; current bounded ridge ablation is complete, family remains open.

## de_empirical_bayes

Estimate effect uncertainty and false-sign risk, using historical folds only. Challenge log expression is not raw count input for DESeq2.

Status: queued_method_review. No measured implementation result yet.

Sources: [Adaptive shrinkage](https://pmc.ncbi.nlm.nih.gov/articles/PMC5379932/) (article and sign-error method discussion)

Next: Specify historical uncertainty estimator, mixture prior and frozen sign thresholds before implementing.

## direction_neural_ode

Nonlinear latent dynamics with a decoder may capture curved development; out-of-range forecasting is a separate assumption.

Status: queued_method_review. No measured implementation result yet.

Sources: [scNODE](https://pubmed.ncbi.nlm.nih.gov/39230694/) (abstract; full PMC access blocked)

Next: Read full methods and author code; assess CPU runtime and count/log likelihood. Earlier linear latent failure does not exhaust neural ODEs.

## distribution_dynamic_ot

Population transport with continuous nonlinear dynamics; interpolation success does not establish extrapolation.

Status: queued_method_review. No measured implementation result yet.

Sources: [TrajectoryNet](https://proceedings.mlr.press/v119/tong20a.html) (abstract and introduction)

Next: Review regularizers, implement past-only rolling temporal folds and compare with linear baseline.

## growth_unbalanced_ot

Growth-aware population evolution can change composition, but captured cell counts do not measure proliferation.

Status: queued_with_input_assumptions. No measured implementation result yet.

Sources: [Waddington-OT](https://pmc.ncbi.nlm.nih.gov/articles/PMC6402800/) (abstract and stability snippets), [TIGON](https://www.nature.com/articles/s42256-023-00763-w) (abstract and method snippets)

Next: Determine defensible growth priors or measurements; do not equate sampled cell counts with biological growth.

## mmd_kernel_matching

Forecast past kernel-embedding drift, then entropy-regularized whole-cell resampling. Original KMM uses observed target covariates; this uses an extrapolated target and is an adaptation.

Status: implemented_evaluated_no_joint_gain. Six variants and two controls across three panels: best variant 52.0672, below unchanged control 55.3123.

Sources: [Kernel two-sample test](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf) (abstract and kernel formulation), [Kernel mean matching](https://papers.nips.cc/paper_files/paper/2006/file/a2186aa7c086b46ad4e8bf81e2a3a19b-Paper.pdf) (quadratic program method), [Random Fourier features](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf) (abstract)

Next: Evaluate actual full-panel metrics; reject surrogate-only improvements. Cannot generate absent cell states.

## mmd_generative_moments

Train a conditional generator using kernel discrepancies between past timepoint populations.

Status: queued_method_review. No measured implementation result yet.

Sources: [Generative moment matching networks](https://proceedings.mlr.press/v37/li15.html) (abstract and introduction)

Next: Read full training method, choose past-only conditional time model and check compute dependencies; T1 adaptation needs separate validation.

## css_covariance_shrinkage

Regularize historical covariance and forecast joint dependence. Static covariance estimation alone does not forecast development.

Status: queued_method_review. No measured implementation result yet.

Sources: [Ledoit-Wolf covariance](https://www.ledoit.net/Well-conditioned2004.pdf) (abstract), [Variogram scoring rules](https://repository.library.noaa.gov/view/noaa/22327/noaa_22327_DS1.pdf) (abstract and correlation experiment discussion)

Next: Review shrinkage formula; implement low-rank PSD covariance dynamics with zero/protected-gene guards and rolling folds.

## css_direct_variogram

Optimize historical expected absolute gene-pair differences to power .5; covariance alone misses non-Gaussian dependence.

Status: implemented_evaluated_no_joint_gain. Six past-pair weighting variants across three panels: best variant 51.4766, below unchanged control 55.3123.

Sources: [Variogram scoring rules](https://repository.library.noaa.gov/view/noaa/22327/noaa_22327_DS1.pdf) (abstract and correlation experiment discussion)

Next: Freeze past-selected pair surrogate and joint-program adjustment; evaluate unchanged full-panel 20k-pair metric.

## css_copula

Separate marginal calibration and joint rank dependence, adapting ensemble methods to gene expression.

Status: queued_method_review. No measured implementation result yet.

Sources: [Ensemble copula coupling](https://arxiv.org/abs/1302.7149) (abstract)

Next: Review full method, handle zero ties, conserved mapped mass and protected genes; do not import future ranks.

## integration_whole_cells

Combine whole-cell specialist forecasts; scores must be evaluated jointly, not added across models.

Status: implemented_evaluated. DE mixture weights .1/.25/.5/.75 tested; .75 raises DE but reduces overall to 54.8128.

Sources: Local integration experiment.

Next: Revisit integration only when new specialists improve the Pareto frontier; other mixtures remain untested.

## Experimental interpretation

Kernel features use a past-fitted 16-dimensional encoder and 500 random features, while the actual MMD evaluator uses its unchanged full-panel scoring procedure. Lower training-surrogate error is not sufficient. Whole-cell resampling preserves expression within selected cells but cannot synthesize unseen populations.

Current best three-panel point mean: reliability expression blend .5, 55.3744. The previous unit_k16 stability pilot remains 55.2813 mean and 52.6016 empirical lower tail over 16 replicates; those bounds do not apply to the new reliability candidate. No >72 readiness gate has passed.

## Completed kernel and variogram batches

Both implemented population-weighting branches failed to improve the unchanged unit_k16 control. The queue now contains complete four-metric results and report hashes for each. Six variants plus two controls across three panels per branch consumed 48 local evaluations, zero submissions and zero Jev calls. These failures do not exhaust nonlinear generative models, empirical Bayes gene calibration or covariance dynamics.

A 16-replicate paired stability pilot is running for reliability expression blend .5 against its unchanged control. Its .0621 point gain must not be treated as confirmed.

The completed reliability pilot rejected promotion: mean 55.1454 versus incumbent 55.2813. Retain unit_k16. Its apparent three-panel point improvement was not robust.
