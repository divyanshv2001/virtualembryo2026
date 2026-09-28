# Prospective task specification

Verified from official public pages on 2026-09-27. These are planning inputs, not evidence of project performance. Released training files require participant access and are absent locally. No registration, download of protected data, training or submission was performed.

| Task | Inputs released during development | Validation | Hidden final test | Prediction |
|---|---|---|---|---|
| T1 | RNA at E8.5/E9.5 | E10.5 | E12.5 | Expression distribution |
| T2 embryo | E6.75/E7.25/E8.0 | E7.5 | E7.75 | Expression and 3D positions |
| T2 heart | E8.25/E8.75/E9.5 | E8.5 interpolation; E10.5 extrapolation | E12.5 extrapolation | Expression and 3D positions |
| T3 | Wild-type series plus Mab21l2 KO E9.5 | Gata4 KO E8.75 | β-catenin KO E8.75 | Expression and 3D positions |

These splits come from [official tasks](https://virtualembryo.ai/challenge/tasks). Public split definitions do not provide hidden values or cell correspondences. Do not describe predicted snapshots as lineage trajectories.

Output is an AnnData-compatible cells-by-genes expression matrix, finite nonnegative log-normalized values, ordered board-specific panel, and spatial_3D coordinates where required. Board panels/limits: T1 32,285 genes and 1,000–5,118 cells; T2 embryo 498 and 583–5,000; T2 heart extrapolation 500 and 1,000–25,179; T2 heart interpolation 500 and 1,000–17,616; T3 500 and 1,000–7,449. Verify the current machine-readable index before writing predictions. Counts are submission sample sizes, not embryo-size predictions. [Official data](https://virtualembryo.ai/challenge/data).

| Task | Official weighted metric groups |
|---|---|
| T1 | DES 25%; DCS 25%; MMD 30%; CSS 20% |
| T2 | Expression change 25% (DES/DCS); state distribution 25% (MMD/CSS); shape and scale 25% (SDD/ODS/TSR); neighborhood fidelity 25% |
| T3 | DES 30%; DCS 25%; response magnitude PSS 25%; state distribution 20% (MMD/CSS) |

DES assesses signed DE recovery after a reference-expression null correction; DCS controls rank association for reference expression. MMD compares populations; CSS compares sampled gene-pair variograms. T2 shape metrics distinguish form and physical scale, while neighborhood fidelity compares neighborhood expression distributions. T3 requires coordinates but does not score tissue geometry. The official evaluator fits its comparison PCA on truth; a learner must never fit preprocessing on hidden truth. Local evaluation may fit a held-out evaluation basis without feeding it back into model training. Floor is copy_last for T1/T2 and wt_identity for T3. [T1 scoring](https://virtualembryo.ai/challenge/evaluation?section=scoring), [T2 scoring](https://virtualembryo.ai/challenge/evaluation?section=scoring&task=2), [T3 scoring](https://virtualembryo.ai/challenge/evaluation?section=scoring&task=3).

Current rules permit disclosed external resources subject to protected staging/genotype windows; a blanket T2/T3 ban in local source is outdated. Record exact stages, alleles, licences and pretraining exposure before use. Stage/genotype filtering is mandatory, including inherited exposure through pretrained models. Development quotas are defined per task/day; final limits differ. Returned scores may select predictions but may not be inverted to reconstruct target properties. This audit uses no score returns or anchor inversion. Agent-track eligibility requires a locked configuration and run evidence; this research audit does not produce an eligible prediction or certify a past run. [Official rules](https://virtualembryo.ai/challenge/rules).

Outstanding contracts: actual downloaded index/code versions, replicate identities, coordinate units, conditional-KO timing/tissue, assay preprocessing, reference-file exact names, organism/sample licences and compute budget. Metric text and baseline descriptions may disagree about structure-gated shape aggregation; obtain a versioned scoring implementation before claiming exact local parity. Missing contracts remain INSUFFICIENT EVIDENCE.
