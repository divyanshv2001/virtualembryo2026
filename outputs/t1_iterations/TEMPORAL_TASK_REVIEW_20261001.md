# Official temporal task review

Reviewed 2026-10-01 after the harness iteration. Sources:

- https://virtualembryo.ai/challenge/tasks/temporal
- https://virtualembryo.ai/challenge/evaluation?section=submissions&task=1
- https://virtualembryo.ai/challenge/rules

Task 1 trains on E8.5/E9.5, predicts leaderboard E10.5 and hidden final E12.5. It evaluates whole-transcriptome cell populations, including new states and proportions, without spatial coordinates. Expression-only shifts may therefore miss population emergence; this is an inference about our failed trend candidates, not a demonstrated diagnosis or a new validated method. Retain the declared observed-only detection calibration test and its matched temporal controls; population birth/state forecasting remains an open path.

Submission contract: finite nonnegative two-dimensional float32 log-normalized .X, exact 32285-gene panel/order and at least 1000 cells; consult the board-specific Data/index.json limits before new exports. Labels do not determine evaluation. The existing 1500-cell export is not improved by this documentation review.

Rules section 10 permits disclosed external public sources for Task 1, including exactly E9.5. Its protected external interval is after E9.5 through E13.5 inclusive; E10.5/E12.5 targets are absolutely excluded. Task 2 embryo E7.5/E7.75 restrictions are task-specific and do not by themselves disqualify earlier-stage external Task 1 training. Stage filters and source disclosure must accompany our method summary. No new data was acquired or used in this review.

Scores may select among predictions but must not be inverted to recover hidden target properties. Agent-track evidence must correspond to the genuine producing run; preserve trajectories, prompts and harness provenance. This new orchestration harness must not be represented as the harness of older submissions. No upload was made.

## Basics audit — 2026-10-04

Rechecked [T1 task](https://virtualembryo.ai/challenge/tasks/temporal), [scoring](https://virtualembryo.ai/challenge/evaluation?section=scoring), and [resources](https://virtualembryo.ai/challenge/evaluation?section=resources). Resources were verified in the browser after HTTP retrieval failed. The official resource package is [veckit](https://github.com/aristoteleo/veckit); its offline results are development evidence, not predictions of hidden-board scores. Do not replace the pinned scorer during ongoing experiments.

| Basic | Audit result |
|---|---|
| Observed training inputs | Real E8.5: 16,787 cells; real E9.5: 17,057 cells; both have the complete ordered 32,285-gene panel. |
| Best export training coverage | External temporal field uses 25,963 cells. Real E9.5 contributes anchors/alignment; the two real observed stages do not jointly supervise that field. Gap. |
| Expression/file contract | Existing best artifact: 1,500 cells, finite nonnegative float32 log-normalised expression, exact gene order; schema verified. |
| Full-transcriptome predictions | External mapping supports 26,775 genes; 5,510 genes retain E9.5 values. Gap. Only four unsupported genes have observed-stage absolute mean changes >=0.25 (55 overall), so this count alone does not explain weak DES. |
| New states and proportions | 18 observed types at E8.5, 21 at E9.5, 11 shared names. Names absent at E8.5 account for 48.05% of E9.5 cells; naming differences do not prove lineage emergence. No validated population-emergence mechanism. Gap. |
| Four-metric implementation | Pinned core DE/null correction, partial-rank direction, target-only PCA/unbiased five-bandwidth MMD, p=0.5 variogram over 20,000 pairs covered. Weights 25/25/30/20 covered. |
| Calibration and temporal validation | Historical full-panel calibration unchanged and valid; local development scores remain distinct from published E10.5 anchors and official results. No independent evidence for scores above 70. |
| Compute and provenance | New potential training used RTX3060; 16GB host cap, D-only data, frozen predictions, matched controls and genuine run evidence preserved. No official upload. |

Next priority: predeclare a real-observed-stage, full-transcriptome training/decoder ablation before another architecture search. Separately test a state-conditional mixture-transition mechanism against frozen copy and CNF controls; do not combine changes before attributing gains. Use permissible earlier-stage development tests, fresh training seeds, all four metrics, and unchanged readiness gates. E10.5/E12.5 targets remain excluded from training. Training coverage and population forecasting are hypotheses to test, not promised score gains.

Chronological potential batch finished: E8.25 mean49.28945 versus CNF49.71122; E8.5 mean49.50182 versus CNF49.50783; copying50 at both. All18 calibrations valid. No promotion; official best remains52.16.
