# Conditional distribution matching experiment

Status: completed, no promotion. The >72 readiness gate remains unmet; challenge incumbent is unchanged.

[Generative Moment Matching Networks](https://proceedings.mlr.press/v37/li15.pdf) trains feedforward generators using kernel discrepancies and also evaluates generation through a learned code space. Its published applications are image datasets, not this T1 forecast. This CPU implementation is a conditional residual adaptation: observed past latent cells plus four noise coordinates enter a 64-unit tanh network; quarter-day residual outputs are trained against the next observed past-stage population. It uses an autonomous repeated transition, not a continuous normalizing flow or a faithful reproduction of the paper architecture. Conditional evolution beyond the observed period remains an assumption.

The learner fits all representation, standardization, bandwidths, network parameters and ridge gene decoder using rows no later than the fold cutoff. The loss is squared biased latent MMD averaged over five past-calibrated RBF scales, plus a small velocity penalty. Eight hundred Adam updates use 64-cell batches. Fixed cell groups drawn from the observed past stages select the checkpoint. They are internal cell validation, not independent temporal validation; future-stage expression is never used for training or checkpoint selection.

Declared forecast variants: strengths .25/.5/1 with fixed-seed noise, and .5 with zero noise, plus persistence and the state-based control. Gene factors are capped at 1.25, mapped gene mass is conserved, missing/ambiguous genes are protected and covariance changes are guarded. The decoder preserves zeros and cannot express novel gene activation. These constraints limit distribution changes and must be retained in interpretation.

The published paper also recommends a square-root loss. The initial experiment uses squared loss; failure does not exhaust that objective, model width, noise dimension, larger training batches, time conditioning or a count-aware decoder. The exact MMD derivative passes a finite-difference check; future-expression mutations leave fitted parameters and forecasts unchanged in the synthetic test, and protected genes/library mass are checked.

Earlier E8.0-to-E8.25 and E8.25-to-E8.5 full-panel evaluations run first. A separate whole-atlas E7.5-to-E8.5 mechanism check uses the existing broader cohort. It does not certify the cardiac-associated challenge configuration. The latent training surrogate never replaces the official-panel four-metric scorer. No submissions or Jev calls.

## Completed squared-loss pilots

Quarter-day MMD-trained forecasts did not generalize consistently. Strength .25 scored 50.5195/46.8922; .5 scored 50.5066/46.5860; 1 scored 50.1761/45.9978; meanflow .5 scored 50.5033/46.5839. The internal past MMD loss fell from .04695 to .01525 at cutoff E8.0 and from .04839 to .01598 at cutoff E8.25, but actual full-panel future metrics did not reliably improve.

One-day E7.5-to-E8.5: persistence 50, state control 51.5766, noisy generator .25 49.0386, .5 48.8819, 1 48.7191, meanflow .5 48.8977. All calibrations valid; no promotion. Training surrogate progress is not forecast performance. E9.5 challenge expression has not been used in these generator runs.

A separately declared square-root training-loss ablation is running with the same architecture, batch size, update budget and evaluation controls. Historical executed code is preserved against each original plan hash before changing the implementation. Both squared and square-root exact gradients are checked numerically; the square-root branch includes a small numerical floor.

## Completed square-root ablation

Quarter-day noisy strengths .25/.5/1 scored 56.0911/46.9228, 56.7443/46.6836 and 57.5974/46.1006 respectively across the two folds. Meanflow .5 scored 56.6937/46.6693. The improvement on the first fold does not transfer to the second.

One-day E7.5-to-E8.5: persistence 50, state control 51.5766, noisy strengths .25/.5/1 scored 49.1287/48.9500/48.5296; meanflow .5 scored 48.8823. All calibrations valid. Neither objective beats the one-day state control or achieves consistent temporal gains. No model promoted.

Thirty-six local candidate evaluations across four runs are complete. Zero challenge E9.5 evaluations, submissions or Jev calls for this experiment. All 45 tests passed. Retain unit_k16 challenge incumbent: 55.2813 stability mean, 52.6016 lower tail. The >72 goal is not achieved.

Next mechanism: explicitly stage-conditioned transitions with past-only sign reliability and count-aware decoding, rather than a single autonomous transition shared across stages. This is a hypothesis inferred from inconsistent fold performance, not established biological causation. Model width, batch size, time conditioning, richer decoding and original published architectures remain unresolved; this bounded adaptation is complete, not the whole research family.
