# Production coverage audit specialist reviews

Specialist role policy v2. Fresh per-metric packets; each review limited to150words. Metrics unavailable throughout this read-only audit; reward0. These reviews are proposals, not verified causal diagnoses or human credentials.

## Failed audit: production_coverage_audit_01

Private report SHA256 8cd4c4fcbcea533fae5504b844c4ece20f114ee6eb4b9dfaa80203ff8614fedf. Four fresh agents completed: production_audit_de, production_audit_direction, production_audit_mmd, production_audit_variogram.

### DE

Problem: FileNotFoundError prevented provenance checks and DE outcomes are unavailable. Proposed solution: locate the preserved executed exporter, use export_formula_progress_submission_run.py, and launch a distinct retry while retaining originals. Validation: verify exact donor sampling, gene ordering, source/plan/artifact hashes and time policy, protected genes and mass. Coordinator correction: audit success does not require DE outcomes; DE is evaluated separately on matched historical holdouts. Readiness unchanged.

### Direction

Problem: Missing exporter filename prevented all checks; empty direction outcomes provide no improvement evidence. Proposed solution: corrected read-only provenance audit, then separate past-only scored validation with identical donors, gene order, preprocessing and sampling. Validation: provenance/time passes first; evaluate signed DE under frozen criteria and uncertainty estimates. Missing scores or unmatched controls invalidate performance claims, not a successfully completed unscored audit.

### MMD

Problem: Audit failure before provenance checks; no distribution score. Proposed solution: distinct filename-corrected retry, followed by matched historical scoring with fixed genes, samples, preprocessing and strata. Validation: successful integrity checks before interpretation; report MMD with uncertainty and prespecified control improvement criteria. Audit completion alone is not predictive readiness.

### Variogram

Problem: Wrong exporter filename prevented verification; no variogram is expected from this audit. Proposed solution: verify corrected provenance, then separately assess nonspatial expression covariance on past-only matched donors and gene order. Validation: identical historical targets, strata and scorer; prespecified lower variogram error and intact provenance. No coordinate assumptions or performance claims.

## Completed retry: production_coverage_audit_retry_01

Private report SHA256 645b665c67e4c4a4c0109bbb20b7dc8e2cce134672309917a48b5d336d4995f2. Four fresh agents: production_retry_de, production_retry_direction, production_retry_mmd, production_retry_variogram.

### DE

Problem: Provenance, ordering, timing, protection and mapped-mass checks pass. Detections lost171559 versus gained71377, netloss100182. Mass conservation does not preserve DE structure. Protected2605 active genes represent1.396% of abundance; low mass does not prove negligible DE effect. Proposed solution: calibrate weaker mapped redistribution on eligible past observations. Validation: matched copy/current/calibrated transforms on one-day historical holdouts, with protection/mass and DE criteria intact. Coordinator correction: reducing detection loss alone is not success; the real historical target may lose detections. Official52.16 attribution is unproven.

### Direction

Problem: Audit passes; direction unavailable. Protected coverage and detection turnover do not establish biological harm. Proposed solution: historical one-day signed DE validation, comparing current update, persistence and damped update; choose damping only on earlier folds. Preserve protected genes and mapped mass; do not force detection recovery. Validation: matched donors, sampling, normalization and seeds, biological replicates where available. Reproducible direction gain over persistence without material regression versus incumbent; inconsistent gains or missing comparable endpoints fail. Audit success stands independently of scores.

### MMD

Problem: MMD unavailable. Timing/hash/order pass; net detection loss need not be harmful, and unmapped low abundance need not be distributionally negligible. Source/challenge difference does not identify temporal error. Proposed solution: same-donor one-day copy/incumbent/candidate comparison with fixed full gene space and bandwidth; diagnose mapped/protected discrepancies alongside detection shifts. Tune shrinkage only on past validation if unsupported transport is worse. Validation: prespecified historical control improvements with uncertainty, provenance/order/time intact. Missing scores leave benefit unestablished.

### Variogram

Problem: Variogram unavailable, so covariance quality and official-score impact unknown. One-day call/hash/order pass. Detection loss and unmapped low abundance do not establish harm or negligible influence. Proposed solution: historical one-day expression-increment discrepancies in actual scoring gene space, stratified by mapped/ambiguous/absent genes and detection; compare identical copy/incumbent donors and preprocessing. If mapped covariance degrades, tune covariance-preserving shrinkage on past holdouts, keeping protected genes. Validation: actual one-day variogram improvement against both controls without material stratum regressions; otherwise reject. Readiness unchanged.

## Coordinator decision

Coverage is measured, but no causal official-score diagnosis yet. Actual one-day time call is correct. Protected active genes are2605/22141, but only1.396% expm1 abundance; prioritize actual one-day decoder error decomposition rather than assuming these genes explain the plateau or blindly preserving detection. Old prediction matrices were temporary; any unique diagnostic reconstruction must exactly reproduce the original hashes before target reads. Jev chose this diagnostic over immediate new training/missing-gene-only/minimum-detection-loss changes:2539byte packet,1116input/73output tokens. No score/reward or official upload. Specification: NEXT_ONE_DAY_DECODER_DIAGNOSTIC.json.
