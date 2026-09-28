# Nonlinear transport and validation audit

The local >72 readiness gate remains unmet. No official submission or Jev request was made.

The past-source/anchor audit confirms the full 32,285-gene panel, 26,775 unique exact shared genes and 5,510 protected genes. Both datasets have per-cell abundance near 10,000 after reversing log1p. Mean absolute source/anchor difference in 384 encoder features is 0.3868 and 83.3% of anchors satisfy the past-source distance guard. Similar library size is not evidence of domain alignment. The source cohort is cardiac plus 18 associated types, selected with atlas metadata. Sample identifiers are not established independent embryo identifiers.

This subset has E7.5, E7.75, E8.0, E8.25 and E8.5 as past stages. With three required training stages, it cannot provide an earlier one-day rolling forecast. Quarter-day rolling tests cannot certify E8.5-to-E9.5 extrapolation. Future expansion toward earlier atlas stages remains a research path.

[TrajectoryNet](https://proceedings.mlr.press/v119/tong20a.html) models continuous transport with neural dynamics; its published experiments address interpolation. The [author implementation](https://github.com/KrishnaswamyLab/TrajectoryNet) uses PyTorch, absent from the current Python environment. The implemented CPU experiment is an adaptation: Sinkhorn barycentric pseudo-velocities from past stage pairs, autonomous tanh MLP, eight-step RK4 and ridge decoding of latent changes into gene log-abundance factors. It does not implement a continuous normalizing flow, growth model, velocity evidence or the complete published loss. Transport pseudo-velocities are assumptions, not measured cell trajectories.

Initial pilot: six configurations across E8.0-to-E8.25 and E8.25-to-E8.5, with all forecasts frozen before each target read. All calibrations valid. Persistence is 50/50; incumbent 45.7315/50.7535; MLP strength .25 is 44.9879/49.2408; .5 is 44.7441/48.3637; 1 is 44.4754/47.4948; ridge velocity .5 is 44.8680/48.3449. No promotion. Both MLP fits reached the 250-iteration cap, so the neural family is not exhausted.

Correction underway: use observed donors as the flow starting distribution and raise the MLP iteration cap to 1,000. The original model, runner and dependency sources are preserved privately with their original plan hashes. The new run has a separate frozen plan; it reuses temporal development and is not a blind test. Only a completed evaluation can justify a subsequent challenge-development run. Missing/ambiguous genes are protected, mapped library mass is conserved and covariance changes are guarded. The decoder cannot activate zeros, so unsupported new expression remains a limitation.

## Completed standalone correction

The higher-budget standalone flow still failed both quarter-day folds. MLP strength .25 scored 49.8302/48.3581; .5 scored 49.1932/47.8228; 1 scored 48.5146/46.9116; ridge .5 scored 49.7771/47.7460. Persistence scored 50/50. All calibrations valid. Network iteration counts: [279, 397]. This branch is not promoted.

An existing whole-atlas prepared sample includes E6.5 onward. A separate one-day E7.5-to-E8.5 mechanism pilot is now running with 750 anchors and disjoint 500-cell truth/ceiling groups. Fit inputs stop at E7.5. Its broader cohort differs from the cardiac-associated challenge subset, so it cannot certify the same-configuration challenge gate. No re-download, new paid calls or submissions.

## Completed one-day mechanism check

E7.5-to-E8.5 results: persistence 50, state-based control 51.5766, standalone MLP .25 50.7563, .5 50.1856, 1 49.5219, ridge .5 49.9669. All calibrations valid. No nonlinear branch beats the state-based control. This completes 30 local candidate evaluations across the three pilot runs. Zero submissions and zero Jev calls. Retain unit_k16 challenge incumbent (55.2813 stability mean, 52.6016 lower tail); the >72 goal is unfinished.

Next viable paths: a generator trained against past distribution-level losses rather than barycentric pseudo-labels; explicit dependence-preserving gene decoding; and historical sign-aware effect shrinkage. The CPU adaptation failed its declared tests, but it did not reproduce the full TrajectoryNet/scNODE objectives. These research families remain open.
