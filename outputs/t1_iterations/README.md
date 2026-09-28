# Controlled T1 development iterations

The user reported 48.1 for the full exploratory per-celltype mean-shift artifact. This is the first measured leaderboard observation; component scores are not yet supplied. It does not establish which metric failed, and no local metric can replace E10.5 assessment.

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
