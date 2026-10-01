# Official temporal task review

Reviewed 2026-10-01 after the harness iteration. Sources:

- https://virtualembryo.ai/challenge/tasks/temporal
- https://virtualembryo.ai/challenge/evaluation?section=submissions&task=1
- https://virtualembryo.ai/challenge/rules

Task 1 trains on E8.5/E9.5, predicts leaderboard E10.5 and hidden final E12.5. It evaluates whole-transcriptome cell populations, including new states and proportions, without spatial coordinates. Expression-only shifts may therefore miss population emergence; this is an inference about our failed trend candidates, not a demonstrated diagnosis or a new validated method. Retain the declared observed-only detection calibration test and its matched temporal controls; population birth/state forecasting remains an open path.

Submission contract: finite nonnegative two-dimensional float32 log-normalized .X, exact 32285-gene panel/order and at least 1000 cells; consult the board-specific Data/index.json limits before new exports. Labels do not determine evaluation. The existing 1500-cell export is not improved by this documentation review.

Rules section 10 permits disclosed external public sources for Task 1, including exactly E9.5. Its protected external interval is after E9.5 through E13.5 inclusive; E10.5/E12.5 targets are absolutely excluded. Task 2 embryo E7.5/E7.75 restrictions are task-specific and do not by themselves disqualify earlier-stage external Task 1 training. Stage filters and source disclosure must accompany our method summary. No new data was acquired or used in this review.

Scores may select among predictions but must not be inverted to recover hidden target properties. Agent-track evidence must correspond to the genuine producing run; preserve trajectories, prompts and harness provenance. This new orchestration harness must not be represented as the harness of older submissions. No upload was made.
