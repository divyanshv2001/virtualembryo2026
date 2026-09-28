# Controlled T1 development iterations

**Persistent objective:** [LOCAL_OPTIMIZATION_PROMPT.md](LOCAL_OPTIMIZATION_PROMPT.md) governs the user-requested local >72 loop, Monte Carlo checks and checkpoint/reset recovery. Read `LOCAL_OPTIMIZATION_STATE.json` to resume. A completed experiment below does not mean the threshold has been achieved. The current prompt generator includes a compact pointer to this mandate rather than duplicating the full research history across experts.

The [prompt update and Monte Carlo pilot](MONTE_CARLO_RESULTS.md) completed 16 fixed-seed resampling replicates. Best frozen candidate mean: 50.92; empirical lower tail: 50.55. The objective remains unmet. Local jobs are finished, and reset timing is unknown because no account-specific limit event/status is exposed to this session.

**Current policy:** preserve the user's four remaining daily submissions. The older upload sequence below is superseded. The executed offline backtest and its limitations are documented in [LOCAL_VALIDATION.md](LOCAL_VALIDATION.md); no local or official score above 72 has been established.

The larger E6.5–E9.5 atlas acquisition and prospective training workflow are described in [EXTENDED_ATLAS.md](EXTENDED_ATLAS.md). Its local gene panel and calibration differ from the official board.

The best user-reported result remains 48.1 for the full exploratory per-celltype mean-shift artifact. No local diagnostic replaces E10.5 assessment. See [OPTIMIZATION.md](OPTIMIZATION.md) for the confirmed feedback and prioritized experiments.

Round 1 freezes a small training-only candidate set before execution:

| Candidate | Change from the same E9.5 donor cells |
|---|---|
| Existing sampled persistence | Unchanged expression; unscored control |
| shrunk_shift_a010 | 10% of the E8.5→E9.5 per-type mean difference, clipped at zero |
| shrunk_shift_a025 | 25% of that difference, clipped at zero |
| zero_preserving_shift_a025 | 25% shift applied only to positive donor entries, with clipping |

These are declared hypotheses, not predicted leaderboard winners. All candidates use identical donor rows/panel/annotations; no target cells, external atlas or score-derived target reconstruction is used. The zero-preserving variant tests support changes introduced by dense additive shifts; real biology need not preserve zeros. No library normalization is inferred. Unmatched annotation labels stay unchanged. Current labels are not biologically harmonized.

## Round 2: empirical distribution reweighting

`composition.py` builds stratified persistence and a capped composition-trend candidate. For exact shared labels, weights are proportional to p_E9.5 × clip((p_E9.5/p_E8.5)^0.5, 0.5, 2); unmatched last-stage labels have multiplier 1. Allocate exact cell quotas by largest remainder, sample without replacement within labels and retain every selected expression row unchanged. Shared RNG settings make this a declared distributional comparison, although donor sets differ when quotas differ. Zero-strength composition weights reproduce empirical proportions up to integer rounding; they are not an official floor-parity claim.

This avoids dense gene shifts while testing whether population weights help. It cannot distinguish dissection/capture changes from real developmental abundance, harmonize labels or generate an unseen future population. The screenshot's leading 71.7 is a user-supplied benchmark target; its method has not been recovered or reproduced. No candidate is claimed to beat it until scored.

```powershell
outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/composition.py --round round_02_composition
outputs/research_workflow/.venv/Scripts/python.exe -m unittest discover -s outputs/t1_iterations -p test_composition.py
```

Keep at least one matched persistence result and the full-shift result for interpretation. Prioritize distinct hypotheses over an indiscriminate parameter sweep. New score feedback must name its artifact before it enters the ledger; ambiguous component lists remain unassigned. More elaborate density-ratio/latent-state models require appropriate training-only validation or eligible independent data, and their superiority cannot be inferred from architectures or local fit alone.

Submit the persistence control first. Obtain its aggregate and DES/DCS/MMD/CSS breakdown. Then test one smaller-shift candidate, retaining negative/ambiguous results and updating the checksum-linked score ledger. Comparing ordinary parameter candidates is permitted; recovering hidden-target properties from returned scores is prohibited by [rules §10](https://virtualembryo.ai/challenge/rules). The screenshot showed 1/8 T1 attempts for that day; recheck actual remaining quota before each upload. This code does not upload.

## Execute and record feedback

```powershell
outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/iterate.py --round round_01
outputs/research_workflow/.venv/Scripts/python.exe -m unittest discover -s outputs/t1_iterations -p test_iteration.py
outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/record_score.py --round round_01 --artifact outputs/t1_run/T1_val__sampled_copy_last.h5ad --score SCORE_FROM_PORTAL
```

Optional `--metrics-json PATH` records numeric component scores. A duplicate artifact/score/metrics observation is not appended twice. Round directories cannot be reused: choose a new name when making a materially different batch. The plan, tool-source hash, input/donor checksums, local execution event log, diagnostics and score ledger stay under ignored `private/`; only reproducible code/docs are committed.

Diagnostics measure clipping, changes to zero support and output magnitude only. They do not rank future accuracy. T1 has just two supplied stages, so a stage holdout cannot validate a two-stage trend. Each official feedback cycle can select parameters without supporting a new biological mechanism.

Jev usage is zero for known deterministic steps. Ambiguous future routing can use compact, separately approved bounded requests; the exhausted three-call audit approval is not extended by this loop. A batch plan is not the complete prompt/model/tools/permissions/budget configuration lock required for Agent Team eligibility. This remains development work; a later eligible autonomous entry needs a properly locked run with matching actual evidence.

## Round 3: expression-state transport

`local_transport.py` fits a label-free, training-only PCA representation using 768 pooled high-variance genes and 24 dimensions. This is a nearest-neighbor transport heuristic, not an optimal-transport solver. It ignores the supplied Harmony embedding. For each of the original 2,000 donor cells, it finds local E8.5/E9.5 neighbors in two disjoint reference halves, excluding donors from E9.5 reference pools. Feature selection and representation are fitted on all training cells; these halves assess conditional neighborhood-estimator stability, not end-to-end held-out generalization.

Local means and standard errors are computed in expm1(X) space, assuming natural-log-normalized input as required by the submission contract; these values are not raw counts. Mean differences determine signal-to-noise shrinkage. A half-strength log abundance ratio is capped at a factor of two, applied only where the two estimates agree in sign, and discounted for distant earlier-stage matches. Mapping back with log1p retains every donor zero. Neighborhood changes can differ across cells and apply to labels without exact earlier-stage matches. Expression similarity does not identify ancestry, correct dissection effects, or prove extrapolation.

```powershell
outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/local_transport.py --round round_03_transport
outputs/research_workflow/.venv/Scripts/python.exe -m unittest discover -s outputs/t1_iterations -p test_transport.py
```

The executed candidate passed local format checks: 2,000 × 32,285, float32, exact panel, finite nonnegative values and no coordinates. Median sign agreement among active genes was 0.635; median distance-based matching trust was 0.755. Neither statistic measures prediction accuracy. Its unchanged zero support also prevents new gene activation, a deliberate limitation to test rather than a biological assertion. Actual plan, representation, execution events and checksum-linked report are private artifacts. The user subsequently supplied scores of 39.6 DE recovery, 52.4 direction, 49.9 MMD and 47.7 variogram: a derived weighted total of 47.51, below the best reported 48.1. A portal headline was not supplied. See OPTIMIZATION.md for the resulting priorities.

## Distribution constraints and rolling forecasts

The latest [distribution audit and rolling results](CONSTRAINED_RESULTS.md) supersede historical upload suggestions above. Two whole-atlas batches completed five time splits each, preserving donor zero support and constraining expression changes. No model passed promotion; submission quota remains unused. Code and results are reproducible, with actual execution histories retained privately.

The subsequent [weak-fold diagnosis and robust forecast results](ROBUST_RESULTS.md) expand the cohort to 25,963 cells and test coarser states, uncertainty shrinkage and covariance bounds. Another 96 model/fold evaluations completed. No model passes the 72 promotion gate; the explicit gene-transfer helper is tested but no challenge file is promoted.

The [full-panel challenge-data backtest](CHALLENGE_BACKTEST_RESULTS.md) now predicts supplied challenge E9.5 from E8.5 anchors with atlas fitting restricted through E8.5. Alignment and direct-projection controls completed 54 candidate/panel evaluations. The best mean local score is 51.08; no E10.5 refit or submission is promoted. Observed-stage backtest artifacts must not be uploaded as E10.5 forecasts.
