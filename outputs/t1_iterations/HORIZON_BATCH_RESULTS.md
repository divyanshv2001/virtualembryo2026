# One-day source-cohort audits

Two observed one-day folds, E8.0→E9.0 and E8.25→E9.25, use the same cardiac-associated source cohort, 1,500 donors and complete official gene ordering. Each forecast is frozen before later-stage expression is read for scoring. Missing/ambiguous genes use zero placeholders in this atlas-only diagnostic; these are not measured challenge genes. These folds therefore improve horizon matching but do not certify challenge-domain or embryo-independent generalization.

The prior model scores 47.143 and 47.987. All component ablations also score below persistence on both folds. This confirms one-day failure within the source cohort; it cannot isolate hidden E10.5 causes.

The positive-quantile mechanism fits three historical positive-expression quantile distributions, suppresses reversing trends, shrinks low-support slopes, and preserves the donor zero mask and number of cells. Per-cell mapped mass and protected genes are guarded. Marginal proposals are monotone; cell-specific mass conservation can change final cross-cell ranks.

A second pilot conditions those margins on four broad past-fitted states. It retains fixed donor counts, changes only trusted anchors, requires 20 anchors and 20 source cells at every recent stage, and applies both within-state and overall covariance guards. It tests composition confounding, not inferred cell proliferation.

The annotation pilots group cells by observed published type strings and estimate positive-abundance and detection trends separately. Four ablations compare expression, detection and their combination. A second frozen pilot raises the source support requirement from 20 to 100 cells per type per past stage and the positive-expression minimum from10 to20. Donor counts stay fixed. Annotations now enter learner grouping; joint atlas annotation is a limitation, and later-stage label values are excluded from fitting.

Shared temporal-profile pilots project gene slopes onto past-only SVD ranks2/4/8 with detection0/.5. These orthogonal factors are denoising statistics, not biologically validated gene programs. A separate cell-level pilot uses NMF ranks8/16, three fixed replicas, 3000 past fit cells, 384 past-selected genes, scaled normalized abundance, median consensus and fixed-component usage fitting. A ridge1 full-panel decoder transfers within-type usage slopes to bounded gene factors. Convergence warnings and consensus dispersion are retained in every model audit.

The initial16-factor cell-level fit hit the200-iteration budget. A separate frozen optimizer repair raises only that budget to800 for rank16, retaining source cells, genes, random seeds, tolerance, decoder and evaluator panels. Higher iteration count is not assumed to improve forecast accuracy.

The2048-gene coverage ablation converged but regressed. Empirical Bayes pilots tested normal-mixture slope shrinkage with error inflation1/2 and sign gates none/.1/.25. Initial EM fits hit1000 iterations; a separate100-step EM plus convex-simplex SLSQP repair converged with scientific settings unchanged. Neither produced meaningful forecast gains.

Covariance pilots used8 past-fitted PCA dimensions and compared OAS with empirical covariance at three strengths. Positive-margin rank coupling used those covariance proposals, fixed zero masks and exact per-type margins before factor clipping and per-cell mass conservation. Projection changes final margins; all actual metrics remain required. Neither branch meaningfully exceeded persistence.

The neural ODE pilot uses a Gaussian latent encoder, reconstruction pretraining and Sinkhorn losses for observed distributions and latent dynamics. Two fixed regularization values and three forecast strengths use all permitted past cells in the prepared cohort as minibatch pools. This is not the complete raw atlas, and stochastic batches do not guarantee every cell was sampled. A past-only full-gene ridge decoder transfers conditional latent-mean drift to bounded donor-relative factors. No type labels enter the neural learner. The final training iteration is fixed before target scoring. Complete forecast metrics, rather than training loss, determine progress.

Author implementation: [scNODE](https://github.com/rsinghlab/scNODE), architecture, solver, training, loss and benchmark sources reviewed and pinned in NEURAL_ODE_AUTHOR_REFERENCE.json. The full paper fetch was blocked. Normalization, velocity and gradient bounds, ridge decoding and conditional-mean forecasting make this an adaptation rather than a reproduction.

A subsequent ablation fits separate detection and positive-abundance heads on frozen beta0 neural latent means, with abundance-only, detection-only and joint forecasts at two strengths. It permits zero-mask changes with common fixed random uniforms, conserves mapped mass and retains protected genes. Clipped linear probability estimates and diagonal-shrunken conditional abundance coefficients are heuristic adaptations. No additional neural dynamics training or target-selected checkpoint is used. These are reused development folds, not fresh blind validation.

A numerical sampling ablation keeps those fitted heads exactly unchanged, verifying every recomputed head array against its archive. Random-order systematic Bernoulli draws constrain each gene's switch count to within one of its expectation before final projection. Within-gene switches become dependent; improved scores or biological covariance are not guaranteed. Independent-draw forecasts are replayed with verified hashes as matched controls. The frozen plan's abundance-only-control phrase refers to the preceding head pilot; the sampling trial directly compares independent and systematic detection-only/joint forecasts.

The growth-proxy branch pins the author mouse marker files and fits historical expression-matched scores. All94 proliferation and187 P53-proxy genes map uniquely. The first audit failed because its expressed-only control filter left insufficient controls for low-expression markers. A separate repair retains zero-expression controls and preserves that failure. The preliminary forecast test resamples complete donor cells using bounded proliferation, negative-P53 or net weights at two strengths, with exact neutral-growth identity and unchanged calibration. This tests composition sensitivity, not an unbalanced transport solver or measured embryonic birth/death rates. Source: [moscot marker implementation](https://github.com/theislab/moscot/blob/440093ccbb8e70de209157d91da839c55b897821/src/moscot/utils/data.py) identifies its mouse apoptosis list as a P53-pathway proxy; [author marginal tutorial](https://moscot.readthedocs.io/en/stable/notebooks/examples/problems/TemporalProblem/800_score_genes_for_marginals.html) discusses prior-growth sensitivity.

Sources: [Stephens supplement](https://stephenslab.uchicago.edu/assets/papers/Stephens2017-supplement.pdf), sectionsS.1-S.2; [Chen et al.](https://arxiv.org/html/0907.4698v1), Gaussian assumptions, OAS derivationIII-C and simulationIV; [sklearn covariance documentation](https://scikit-learn.org/stable/modules/covariance.html). These motivate estimator adaptations and do not establish developmental forecasting accuracy.

Literature: [Schefzik, Thorarinsdottir and Gneiting (2013)](https://arxiv.org/html/1302.7149v2), methods 4.1–4.3 and experiments 5.4 read. ECC separates marginal calibration from rank dependence. Its weather experiments found benefits dependent on the dependence structure. This implementation adapts that separation to historical positive scRNA margins; it is neither faithful ECC nor evidence that ECC improves this challenge.

The [muscat paper](https://www.nature.com/articles/s41467-020-19894-4) motivates separating within-subpopulation state changes from differential abundance. Introduction, simulation results (20–400 cells) and simulation-preprocessing methods were read. It found sizable detection gains between20 and100 cells per subpopulation/sample. It uses replicated samples and count-based inference; our normalized-abundance trend forecast does not reproduce those methods, biological replication or their inferential guarantees.

[Kotliar et al. (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6639075/) distinguishes identity and activity programs and warns that type averages can miss activity and statistical factors need not be biological programs. Introduction, simulation benchmark, preprocessing and consensus methods were read. The cell-level adaptation uses fewer genes/replicas, normalized abundance, no component outlier filtering, and added ridge decoding and temporal extrapolation; it is not cNMF reproduction or a test of the paper's biological claims.

Historical transport fits past-only PCA representations and quarter-day couplings, then extrapolates barycentric velocities with a ridge affine field. Balanced transport and three unbalanced prior settings use the same frozen full-panel scoring. All first-pilot candidates fail persistence on both folds. The entropy convention is explicit and differs from newer POT KL-reference defaults; this is an adaptation, not WOT/moscot reproduction. Couplings use256 cells per stage and PCA uses3000 fit cells, so these are bounded computational pilots, not full raw-atlas transport.

A frozen-flow decoder ablation verifies every archived head array, then separates positive abundance, detection switches and systematic joint sampling. Abundance-only improves over joint forecasts on both development folds, but its best mean50.0276 combines51.7018 and48.3534 and fails consistency. This motivates a declared full conditional latent covariance ridge ablation; model fitting remains past-only. All folds are exposed development data, not untouched validation.

| Run | Candidate | Fold 1 | Fold 2 | Mean |
| --- | --- | ---: | ---: | ---: |
| matched_horizon_audit_01 | copy | 50.000 | 50.000 | 50.000 |
| matched_horizon_audit_01 | unit16 | 47.143 | 47.987 | 47.565 |
| matched_horizon_audit_01 | expression_only | 49.122 | 47.039 | 48.081 |
| matched_horizon_audit_01 | population_only | 47.256 | 48.903 | 48.079 |
| matched_horizon_audit_01 | population_detection | 47.126 | 48.153 | 47.640 |
| quantile_horizon_pilot_01 | copy | 50.000 | 50.000 | 50.000 |
| quantile_horizon_pilot_01 | unit16 | 47.143 | 47.987 | 47.565 |
| quantile_horizon_pilot_01 | quantile_0.25 | 50.975 | 49.091 | 50.033 |
| quantile_horizon_pilot_01 | quantile_0.5 | 50.854 | 48.980 | 49.917 |
| quantile_horizon_pilot_01 | quantile_1.0 | 50.861 | 48.762 | 49.811 |
| state_quantile_horizon_pilot_01 | copy | 50.000 | 50.000 | 50.000 |
| state_quantile_horizon_pilot_01 | unit16 | 47.143 | 47.987 | 47.565 |
| state_quantile_horizon_pilot_01 | quantile_0.25 | 50.592 | 48.203 | 49.398 |
| state_quantile_horizon_pilot_01 | quantile_0.5 | 50.592 | 48.072 | 49.332 |
| state_quantile_horizon_pilot_01 | quantile_1.0 | 50.596 | 47.987 | 49.292 |
| annotation_horizon_pilot_01 | copy | 50.000 | 50.000 | 50.000 |
| annotation_horizon_pilot_01 | unit16 | 47.143 | 47.987 | 47.565 |
| annotation_horizon_pilot_01 | annotation_expression | 49.595 | 47.855 | 48.725 |
| annotation_horizon_pilot_01 | annotation_detection | 48.525 | 45.679 | 47.102 |
| annotation_horizon_pilot_01 | annotation_combined | 49.336 | 46.920 | 48.128 |
| annotation_horizon_pilot_01 | annotation_detection_full | 49.311 | 47.038 | 48.175 |
| annotation_support_pilot_01 | copy | 50.000 | 50.000 | 50.000 |
| annotation_support_pilot_01 | unit16 | 47.143 | 47.987 | 47.565 |
| annotation_support_pilot_01 | annotation_expression | 49.325 | 47.822 | 48.574 |
| annotation_support_pilot_01 | annotation_detection | 48.895 | 45.923 | 47.409 |
| annotation_support_pilot_01 | annotation_combined | 49.345 | 46.998 | 48.172 |
| annotation_support_pilot_01 | annotation_detection_full | 49.448 | 47.122 | 48.285 |
| program_horizon_pilot_01 | copy | 50.000 | 50.000 | 50.000 |
| program_horizon_pilot_01 | unit16 | 47.143 | 47.987 | 47.565 |
| program_horizon_pilot_01 | program_r2_d0.0 | 50.005 | 45.923 | 47.964 |
| program_horizon_pilot_01 | program_r2_d0.5 | 49.849 | 45.483 | 47.666 |
| program_horizon_pilot_01 | program_r4_d0.0 | 49.725 | 46.535 | 48.130 |
| program_horizon_pilot_01 | program_r4_d0.5 | 49.684 | 46.099 | 47.892 |
| program_horizon_pilot_01 | program_r8_d0.0 | 49.940 | 46.757 | 48.349 |
| program_horizon_pilot_01 | program_r8_d0.5 | 49.818 | 46.376 | 48.097 |
| cell_program_horizon_pilot_01 | copy | 50.000 | 50.000 | 50.000 |
| cell_program_horizon_pilot_01 | unit16 | 47.143 | 47.987 | 47.565 |
| cell_program_horizon_pilot_01 | cell_r8_d0.0 | 49.200 | 51.366 | 50.283 |
| cell_program_horizon_pilot_01 | cell_r8_d0.5 | 49.175 | 50.160 | 49.668 |
| cell_program_horizon_pilot_01 | cell_r16_d0.0 | 50.192 | 47.166 | 48.679 |
| cell_program_horizon_pilot_01 | cell_r16_d0.5 | 49.703 | 46.489 | 48.096 |
| cell_program_repair_01 | copy | 50.000 | 50.000 | 50.000 |
| cell_program_repair_01 | unit16 | 47.143 | 47.987 | 47.565 |
| cell_program_repair_01 | cell_r16_d0.0 | 50.403 | 47.206 | 48.804 |
| cell_program_repair_01 | cell_r16_d0.5 | 49.893 | 46.482 | 48.187 |
| cell_program_feature_pilot_01 | copy | 50.000 | 50.000 | 50.000 |
| cell_program_feature_pilot_01 | unit16 | 47.143 | 47.987 | 47.565 |
| cell_program_feature_pilot_01 | saved384_d0.0 | 49.200 | 51.366 | 50.283 |
| cell_program_feature_pilot_01 | saved384_d0.5 | 49.175 | 50.160 | 49.668 |
| cell_program_feature_pilot_01 | genes2048_d0.0 | 47.825 | 50.039 | 48.932 |
| cell_program_feature_pilot_01 | genes2048_d0.5 | 48.422 | 46.247 | 47.334 |
| empirical_bayes_horizon_01 | copy | 50.000 | 50.000 | 50.000 |
| empirical_bayes_horizon_01 | unit16 | 47.143 | 47.987 | 47.565 |
| empirical_bayes_horizon_01 | saved384_d0.0 | 49.200 | 51.366 | 50.283 |
| empirical_bayes_horizon_01 | eb_e1.0_qnone | 49.176 | 46.694 | 47.935 |
| empirical_bayes_horizon_01 | eb_e1.0_q0.1 | 49.289 | 48.723 | 49.006 |
| empirical_bayes_horizon_01 | eb_e1.0_q0.25 | 49.211 | 47.784 | 48.497 |
| empirical_bayes_horizon_01 | eb_e2.0_qnone | 49.325 | 50.009 | 49.667 |
| empirical_bayes_horizon_01 | eb_e2.0_q0.1 | 50.005 | 50.008 | 50.007 |
| empirical_bayes_horizon_01 | eb_e2.0_q0.25 | 50.004 | 50.008 | 50.006 |
| covariance_horizon_01 | copy | 50.000 | 50.000 | 50.000 |
| covariance_horizon_01 | unit16 | 47.143 | 47.987 | 47.565 |
| covariance_horizon_01 | saved384_d0.0 | 49.200 | 51.366 | 50.283 |
| covariance_horizon_01 | cov_oas_s0.25 | 50.007 | 50.000 | 50.003 |
| covariance_horizon_01 | cov_oas_s0.5 | 50.014 | 50.006 | 50.010 |
| covariance_horizon_01 | cov_oas_s1.0 | 50.031 | 50.020 | 50.026 |
| covariance_horizon_01 | cov_empirical_s0.25 | 50.007 | 50.000 | 50.003 |
| covariance_horizon_01 | cov_empirical_s0.5 | 50.014 | 50.006 | 50.010 |
| covariance_horizon_01 | cov_empirical_s1.0 | 50.031 | 50.021 | 50.026 |
| empirical_bayes_repair_01 | copy | 50.000 | 50.000 | 50.000 |
| empirical_bayes_repair_01 | unit16 | 47.143 | 47.987 | 47.565 |
| empirical_bayes_repair_01 | saved384_d0.0 | 49.200 | 51.366 | 50.283 |
| empirical_bayes_repair_01 | eb_e1.0_qnone | 49.176 | 46.693 | 47.935 |
| empirical_bayes_repair_01 | eb_e1.0_q0.1 | 49.287 | 48.839 | 49.063 |
| empirical_bayes_repair_01 | eb_e1.0_q0.25 | 49.211 | 48.187 | 48.699 |
| empirical_bayes_repair_01 | eb_e2.0_qnone | 49.088 | 50.010 | 49.549 |
| empirical_bayes_repair_01 | eb_e2.0_q0.1 | 50.005 | 50.008 | 50.007 |
| empirical_bayes_repair_01 | eb_e2.0_q0.25 | 50.004 | 50.008 | 50.006 |
| copula_horizon_01 | copy | 50.000 | 50.000 | 50.000 |
| copula_horizon_01 | unit16 | 47.143 | 47.987 | 47.565 |
| copula_horizon_01 | saved384_d0.0 | 49.200 | 51.366 | 50.283 |
| copula_horizon_01 | copula_oas_s0.25 | 50.003 | 49.998 | 50.000 |
| copula_horizon_01 | copula_oas_s0.5 | 50.007 | 49.997 | 50.002 |
| copula_horizon_01 | copula_oas_s1.0 | 50.016 | 49.996 | 50.006 |
| neural_ode_horizon_01 | copy | 50.000 | 50.000 | 50.000 |
| neural_ode_horizon_01 | unit16 | 47.143 | 47.987 | 47.565 |
| neural_ode_horizon_01 | saved384_d0.0 | 49.200 | 51.366 | 50.283 |
| neural_ode_horizon_01 | ode_b0.0_s0.25 | 48.380 | 50.237 | 49.308 |
| neural_ode_horizon_01 | ode_b0.0_s0.5 | 48.413 | 50.797 | 49.605 |
| neural_ode_horizon_01 | ode_b0.0_s1.0 | 48.438 | 51.342 | 49.890 |
| neural_ode_horizon_01 | ode_b1.0_s0.25 | 48.536 | 49.535 | 49.036 |
| neural_ode_horizon_01 | ode_b1.0_s0.5 | 48.275 | 49.883 | 49.079 |
| neural_ode_horizon_01 | ode_b1.0_s1.0 | 47.827 | 50.411 | 49.119 |
| neural_hurdle_horizon_01 | copy | 50.000 | 50.000 | 50.000 |
| neural_hurdle_horizon_01 | unit16 | 47.143 | 47.987 | 47.565 |
| neural_hurdle_horizon_01 | saved384_d0.0 | 49.200 | 51.366 | 50.283 |
| neural_hurdle_horizon_01 | saved_ode | 48.438 | 51.342 | 49.890 |
| neural_hurdle_horizon_01 | hurdle_abundance_s0.5 | 49.488 | 50.153 | 49.820 |
| neural_hurdle_horizon_01 | hurdle_abundance_s1.0 | 49.549 | 50.153 | 49.851 |
| neural_hurdle_horizon_01 | hurdle_detection_s0.5 | 48.639 | 48.404 | 48.521 |
| neural_hurdle_horizon_01 | hurdle_detection_s1.0 | 48.623 | 48.386 | 48.504 |
| neural_hurdle_horizon_01 | hurdle_joint_s0.5 | 49.948 | 49.924 | 49.936 |
| neural_hurdle_horizon_01 | hurdle_joint_s1.0 | 49.858 | 49.924 | 49.891 |
| neural_sampling_horizon_01 | copy | 50.000 | 50.000 | 50.000 |
| neural_sampling_horizon_01 | unit16 | 47.143 | 47.987 | 47.565 |
| neural_sampling_horizon_01 | saved384_d0.0 | 49.200 | 51.366 | 50.283 |
| neural_sampling_horizon_01 | saved_ode | 48.438 | 51.342 | 49.890 |
| neural_sampling_horizon_01 | independent_detection_s0.5 | 48.639 | 48.404 | 48.521 |
| neural_sampling_horizon_01 | independent_detection_s1.0 | 48.623 | 48.386 | 48.504 |
| neural_sampling_horizon_01 | independent_joint_s0.5 | 49.948 | 49.924 | 49.936 |
| neural_sampling_horizon_01 | independent_joint_s1.0 | 49.858 | 49.924 | 49.891 |
| neural_sampling_horizon_01 | systematic_detection_s0.5 | 48.692 | 48.508 | 48.600 |
| neural_sampling_horizon_01 | systematic_detection_s1.0 | 48.663 | 48.624 | 48.643 |
| neural_sampling_horizon_01 | systematic_joint_s0.5 | 49.935 | 49.943 | 49.939 |
| neural_sampling_horizon_01 | systematic_joint_s1.0 | 49.910 | 49.943 | 49.927 |
| growth_composition_horizon_01 | copy | 50.000 | 50.000 | 50.000 |
| growth_composition_horizon_01 | neutral | 50.000 | 50.000 | 50.000 |
| growth_composition_horizon_01 | unit16 | 47.143 | 47.987 | 47.565 |
| growth_composition_horizon_01 | saved384_d0.0 | 49.200 | 51.366 | 50.283 |
| growth_composition_horizon_01 | growth_net_p0.5 | 48.023 | 47.705 | 47.864 |
| growth_composition_horizon_01 | growth_net_p1.0 | 48.283 | 47.683 | 47.983 |
| growth_composition_horizon_01 | growth_proliferation_p0.5 | 48.431 | 47.861 | 48.146 |
| growth_composition_horizon_01 | growth_proliferation_p1.0 | 48.003 | 47.849 | 47.926 |
| growth_composition_horizon_01 | growth_p53_p0.5 | 48.297 | 47.137 | 47.717 |
| growth_composition_horizon_01 | growth_p53_p1.0 | 48.671 | 47.962 | 48.316 |
| transport_hurdle_horizon_01 | copy | 50.000 | 50.000 | 50.000 |
| transport_hurdle_horizon_01 | unit16 | 47.143 | 47.987 | 47.565 |
| transport_hurdle_horizon_01 | saved384_d0.0 | 49.200 | 51.366 | 50.283 |
| transport_hurdle_horizon_01 | saved_ode | 48.438 | 51.342 | 49.890 |
| transport_hurdle_horizon_01 | transport_balanced_s0.5 | 49.374 | 46.275 | 47.824 |
| transport_hurdle_horizon_01 | transport_balanced_s1.0 | 48.773 | 45.744 | 47.258 |
| transport_hurdle_horizon_01 | transport_neutral_s0.5 | 49.783 | 47.211 | 48.497 |
| transport_hurdle_horizon_01 | transport_neutral_s1.0 | 49.425 | 46.854 | 48.139 |
| transport_hurdle_horizon_01 | transport_growth05_s0.5 | 49.890 | 47.360 | 48.625 |
| transport_hurdle_horizon_01 | transport_growth05_s1.0 | 49.520 | 46.976 | 48.248 |
| transport_hurdle_horizon_01 | transport_growth1_s0.5 | 49.974 | 47.508 | 48.741 |
| transport_hurdle_horizon_01 | transport_growth1_s1.0 | 49.680 | 47.083 | 48.382 |
| transport_decoder_horizon_01 | copy | 50.000 | 50.000 | 50.000 |
| transport_decoder_horizon_01 | unit16 | 47.143 | 47.987 | 47.565 |
| transport_decoder_horizon_01 | saved384_d0.0 | 49.200 | 51.366 | 50.283 |
| transport_decoder_horizon_01 | transport_balanced_s0.5 | 49.374 | 46.275 | 47.824 |
| transport_decoder_horizon_01 | transport_neutral_s0.5 | 49.783 | 47.211 | 48.497 |
| transport_decoder_horizon_01 | transport_growth1_s0.5 | 49.974 | 47.508 | 48.741 |
| transport_decoder_horizon_01 | decoder_balanced_abundance | 51.523 | 47.252 | 49.387 |
| transport_decoder_horizon_01 | decoder_balanced_detection | 48.020 | 46.066 | 47.043 |
| transport_decoder_horizon_01 | decoder_balanced_systematic | 49.286 | 46.292 | 47.789 |
| transport_decoder_horizon_01 | decoder_neutral_abundance | 51.519 | 48.105 | 49.812 |
| transport_decoder_horizon_01 | decoder_neutral_detection | 48.424 | 46.108 | 47.266 |
| transport_decoder_horizon_01 | decoder_neutral_systematic | 49.875 | 47.288 | 48.582 |
| transport_decoder_horizon_01 | decoder_growth1_abundance | 51.702 | 48.353 | 50.028 |
| transport_decoder_horizon_01 | decoder_growth1_detection | 48.586 | 46.320 | 47.453 |
| transport_decoder_horizon_01 | decoder_growth1_systematic | 50.099 | 47.498 | 48.799 |
| transport_conditional_horizon_01 | copy | 50.000 | 50.000 | 50.000 |
| transport_conditional_horizon_01 | unit16 | 47.143 | 47.987 | 47.565 |
| transport_conditional_horizon_01 | saved384_d0.0 | 49.200 | 51.366 | 50.283 |
| transport_conditional_horizon_01 | decoder_balanced_abundance | 51.523 | 47.252 | 49.387 |
| transport_conditional_horizon_01 | decoder_neutral_abundance | 51.519 | 48.105 | 49.812 |
| transport_conditional_horizon_01 | decoder_growth1_abundance | 51.702 | 48.353 | 50.028 |
| transport_conditional_horizon_01 | conditional_balanced_s0.5 | 51.747 | 47.309 | 49.528 |
| transport_conditional_horizon_01 | conditional_balanced_s1.0 | 51.531 | 47.130 | 49.330 |
| transport_conditional_horizon_01 | conditional_neutral_s0.5 | 51.780 | 48.109 | 49.944 |
| transport_conditional_horizon_01 | conditional_neutral_s1.0 | 51.644 | 48.027 | 49.835 |
| transport_conditional_horizon_01 | conditional_growth1_s0.5 | 51.928 | 48.416 | 50.172 |
| transport_conditional_horizon_01 | conditional_growth1_s1.0 | 51.824 | 48.372 | 50.098 |
| stage_transport_horizon_01 | copy | 50.000 | 50.000 | 50.000 |
| stage_transport_horizon_01 | unit16 | 47.143 | 47.987 | 47.565 |
| stage_transport_horizon_01 | saved384_d0.0 | 49.200 | 51.366 | 50.283 |
| stage_transport_horizon_01 | conditional_neutral_s0.5 | 51.780 | 48.109 | 49.944 |
| stage_transport_horizon_01 | conditional_growth1_s0.5 | 51.928 | 48.416 | 50.172 |
| stage_transport_horizon_01 | stage_neutral_r1.0_s0.5 | 48.328 | 47.362 | 47.845 |
| stage_transport_horizon_01 | stage_neutral_r1.0_s1.0 | 48.024 | 47.169 | 47.597 |
| stage_transport_horizon_01 | stage_neutral_r10.0_s0.5 | 51.286 | 47.983 | 49.634 |
| stage_transport_horizon_01 | stage_neutral_r10.0_s1.0 | 51.154 | 47.886 | 49.520 |
| stage_transport_horizon_01 | stage_growth1_r1.0_s0.5 | 48.550 | 47.562 | 48.056 |
| stage_transport_horizon_01 | stage_growth1_r1.0_s1.0 | 48.275 | 47.400 | 47.838 |
| stage_transport_horizon_01 | stage_growth1_r10.0_s0.5 | 51.402 | 48.240 | 49.821 |
| stage_transport_horizon_01 | stage_growth1_r10.0_s1.0 | 51.275 | 48.163 | 49.719 |
| resolution_transport_horizon_01 | copy | 50.000 | 50.000 | 50.000 |
| resolution_transport_horizon_01 | unit16 | 47.143 | 47.987 | 47.565 |
| resolution_transport_horizon_01 | saved384_d0.0 | 49.200 | 51.366 | 50.283 |
| resolution_transport_horizon_01 | conditional_neutral_s0.5 | 51.780 | 48.109 | 49.944 |
| resolution_transport_horizon_01 | conditional_growth1_s0.5 | 51.928 | 48.416 | 50.172 |
| resolution_transport_horizon_01 | resolution_neutral_cells512 | 52.085 | 48.718 | 50.402 |
| resolution_transport_horizon_01 | resolution_neutral_cells1024 | 52.111 | 48.317 | 50.214 |
| resolution_transport_horizon_01 | resolution_neutral_pcaall | 51.770 | 48.154 | 49.962 |
| resolution_transport_horizon_01 | resolution_growth1_cells512 | 52.097 | 48.837 | 50.467 |
| resolution_transport_horizon_01 | resolution_growth1_cells1024 | 52.214 | 48.648 | 50.431 |
| resolution_transport_horizon_01 | resolution_growth1_pcaall | 51.900 | 48.373 | 50.137 |
| dimensional_transport_horizon_01 | copy | 50.000 | 50.000 | 50.000 |
| dimensional_transport_horizon_01 | unit16 | 47.143 | 47.987 | 47.565 |
| dimensional_transport_horizon_01 | saved384_d0.0 | 49.200 | 51.366 | 50.283 |
| dimensional_transport_horizon_01 | resolution_neutral_cells512 | 52.085 | 48.718 | 50.402 |
| dimensional_transport_horizon_01 | resolution_growth1_cells512 | 52.097 | 48.837 | 50.467 |
| dimensional_transport_horizon_01 | dimension_neutral_d5 | 49.876 | 51.001 | 50.438 |
| dimensional_transport_horizon_01 | dimension_neutral_d8 | 52.929 | 50.495 | 51.712 |
| dimensional_transport_horizon_01 | dimension_neutral_d24 | 52.002 | 48.846 | 50.424 |
| dimensional_transport_horizon_01 | dimension_growth1_d5 | 49.872 | 50.839 | 50.356 |
| dimensional_transport_horizon_01 | dimension_growth1_d8 | 52.766 | 50.476 | 51.621 |
| dimensional_transport_horizon_01 | dimension_growth1_d24 | 52.021 | 49.099 | 50.560 |
| strength_transport_horizon_01 | copy | 50.000 | 50.000 | 50.000 |
| strength_transport_horizon_01 | unit16 | 47.143 | 47.987 | 47.565 |
| strength_transport_horizon_01 | saved384_d0.0 | 49.200 | 51.366 | 50.283 |
| strength_transport_horizon_01 | dimension_neutral_d8 | 52.929 | 50.495 | 51.712 |
| strength_transport_horizon_01 | dimension_growth1_d8 | 52.766 | 50.476 | 51.621 |
| strength_transport_horizon_01 | strength_neutral_s0.25 | 52.688 | 50.377 | 51.533 |
| strength_transport_horizon_01 | strength_neutral_s1.0 | 52.711 | 50.633 | 51.672 |
| strength_transport_horizon_01 | strength_neutral_s2.0 | 52.711 | 50.700 | 51.706 |
| strength_transport_horizon_01 | strength_growth1_s0.25 | 52.608 | 50.334 | 51.471 |
| strength_transport_horizon_01 | strength_growth1_s1.0 | 52.653 | 50.619 | 51.636 |
| strength_transport_horizon_01 | strength_growth1_s2.0 | 52.653 | 50.683 | 51.668 |
| ridge_transport_horizon_01 | copy | 50.000 | 50.000 | 50.000 |
| ridge_transport_horizon_01 | unit16 | 47.143 | 47.987 | 47.565 |
| ridge_transport_horizon_01 | saved384_d0.0 | 49.200 | 51.366 | 50.283 |
| ridge_transport_horizon_01 | dimension_neutral_d8 | 52.929 | 50.495 | 51.712 |
| ridge_transport_horizon_01 | dimension_growth1_d8 | 52.766 | 50.476 | 51.621 |
| ridge_transport_horizon_01 | ridge_neutral_r0.1 | 53.007 | 50.433 | 51.720 |
| ridge_transport_horizon_01 | ridge_neutral_r0.3 | 52.999 | 50.446 | 51.722 |
| ridge_transport_horizon_01 | ridge_neutral_r3.0 | 52.729 | 50.550 | 51.640 |
| ridge_transport_horizon_01 | ridge_growth1_r0.1 | 52.860 | 50.514 | 51.687 |
| ridge_transport_horizon_01 | ridge_growth1_r0.3 | 52.796 | 50.521 | 51.658 |
| ridge_transport_horizon_01 | ridge_growth1_r3.0 | 52.576 | 50.458 | 51.517 |

Status: completed. Full four-metric vectors, raw metrics, model audits and provenance hashes are in HORIZON_BATCH_RESULTS.json.
No new prospective export or official submission. The >72 objective remains unfinished.
