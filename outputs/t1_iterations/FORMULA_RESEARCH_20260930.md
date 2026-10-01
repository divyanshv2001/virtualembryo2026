# T1 formula search — 2026-09-30

Three requested specialist agents independently reviewed the scorer, developmental biology, and transport model. They found **no validated universal closed-form shortcut** analogous to a turbidity calibration law. The local scorer's DE recovery uses signed top-ranked gene overlap; multiplying every predicted change by one positive scalar cannot change that rank. Direction is a partial rank correlation controlling reference expression. Any large gain needs better gene identities, signs, cell distributions or composition, measured under the unchanged full-panel scorer.

## Testable formulas

1. **Soft empirical-Bayes gene slope.** For each mapped gene, estimate recent past-source slope \(\hat b_g\), uncertainty \(s_g^2\), and prior variance \(\tau^2\). Set posterior \(m_g=\tau^2\hat b_g/(\tau^2+s_g^2)\), sign certainty \(q_g=|2\Phi(m_g/\sigma_{g,post})-1|\), then blend \(d'_g=(1-\alpha)d_g+\alpha h m_g q_g^\gamma\). A historical source-only rank screen is frozen with \(\alpha\in\{.25,.5\}\), \(\gamma\in\{0,1\}\), two rolling folds and recent linear slope control. This is an adaptation of shrinkage concepts, not a reproduction of [adaptive shrinkage](https://pmc.ncbi.nlm.nih.gov/articles/PMC5379932/) or [DESeq2](https://pmc.ncbi.nlm.nih.gov/articles/PMC4302049/). Prior hard sign gates and rank trends were weak, so reject promptly if both historical folds do not improve.
2. **Covariate-adjusted hurdle residual drift.** Fit past stage-specific residual intercepts after removing frozen 8D latent-state effects: \(r^D_{g,t}=E_t[1_{X_g>0}-p_g(Z)]\), and analogously \(r^+_{g,t}=E_t[\log(1+X_g)-m_g(Z)\mid X_g>0]\). Shrunk past-only slopes \(b_g^D,b_g^+\) would add \(\lambda h b_g^D\) to detection probability and \(\lambda h b_g^+\) to positive-log change before the existing mass/covariance guards. The [MAST hurdle model](https://pmc.ncbi.nlm.nih.gov/articles/PMC4676162/) supports the detection/positive decomposition; this temporal residual extrapolation is our unvalidated adaptation. Changing tissue composition and capture depth can mimic drift.
3. **Lineage detection odds.** For a supported lineage \(\ell\), \(\hat p_{g\ell}(t+h)=\sigma(\operatorname{logit}p_{g\ell}(t)+h b_{g\ell})\), with conditional-positive mean \(q_{g\ell}\) and expected abundance \(p q\). Earlier [mouse gastrulation single-cell stages](https://www.nature.com/articles/s41586-019-0933-9) and [lineage-specific smooth trends](https://www.nature.com/articles/s41467-020-14766-3) motivate testing, while annotation mismatch and capture-depth confounding are substantial. No claim that those papers establish one-day extrapolation.

Population growth via \(p_\ell(t+h)=\operatorname{softmax}(\log p_\ell(t)+h r_\ell)\) is lower priority: sampling proportions do not directly identify proliferation, and related local composition tests failed. RNA velocity \(ds/dt=\beta u-\gamma s\) is not executable from the released log-normalized matrix without spliced/unspliced layers.

## Jev advisory accounting

- Formula family packet: 2,168 bytes; 815 input/71 output tokens; advised empirical-Bayes gene effects, confidence 0.56.
- Formula-specific packet: 2,024 bytes; 730 input/69 output tokens; advised soft empirical Bayes, confidence 0.58.

Both sent compact summary statistics and choices only. They are routing advice, not measured score gains or demonstrated Codex-credit savings. No datasets, source files, protected targets or secrets were sent. Distinct packets and validated responses are retained in `JEV_FORMULA_PACKET_20260930_01.json`, `JEV_FORMULA_ROUTE_20260930_01.json`, `JEV_FORMULA_PACKET_20260930_02.json` and `JEV_FORMULA_ROUTE_20260930_02.json`.

## Frozen next screen

`eb_gene_slope_screen.py` evaluates the soft EB formula on source 8.0→8.25 and 8.25→8.5, fitting each using only stages at or before its cutoff. This screen uses mapped-gene signed top-200 overlap and partial Spearman as **proxies**, not the official or full-panel local score. Advance to costly prediction generation only if a setting improves chance-adjusted overlap on both folds without direction regression. Preserve all trial outcomes and continue other declared paths if it fails. The >72, 64-replicate and temporal gates remain unchanged.


## Completed screen and export

The frozen two-fold EB signed-rank/direction proxy screen failed; no challenge promotion. User-requested prospective E10.5 file generated with covariance .25 and full E9.5 anchor refit; local format passed. Prior development mean55.98883 does not establish its official score or >72 readiness. Four metric critiques and next hypotheses recorded in METRIC_CRITIQUES_20261001.md.
