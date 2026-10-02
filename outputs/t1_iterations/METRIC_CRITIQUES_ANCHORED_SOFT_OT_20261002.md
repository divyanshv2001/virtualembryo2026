# Anchored soft-OT metric specialist reviews

Report SHA256: 4e35d7f3ee2bc8f2cd96124d8ac0eafe26a8f05b1b3cdc692cc243be38accb70; specialist role-policy version2. Four fresh agents; coordinator compact transcription; proposals remain unvalidated.

## de_score

Problem: DE skill ties the incumbent on all three panels; direction gains do not show improved DE recovery. Reused panels lack independent embryo replication.
Proposed solution: One joint representation/dynamics trial with past-only dynamic regularization, versus identical architecture without that regularizer. Match seeds, budget, preprocessing, genes, thresholds and sampling; retain incumbent reference and stop OT tuning.
Validation: Lock recipe before 9.5 evaluation. Report paired DE skill, precision/recall and recovered-gene counts. Require positive mean DE gain and no panel regression; a tie or gain explained solely by representation fails the regularization hypothesis. Seed variability is computational stability.

## de_direction

Problem: Direction improves on all three panels, mean +.004203, but aggregate declines, DE ties and MMD/CSS worsen. No promotion or biological generalization claim.
Proposed solution: Stop OT grid. Test scNODE-inspired jointly trained representation/dynamics with past-only regularization and otherwise identical zero-regularizer control, matching seeds, budget and full gene output.
Validation: Freeze recipe before 9.5. Require paired direction improvement over matched controls while aggregate and other metrics satisfy frozen guardrails. Reject gains dependent on one panel or violating guardrails; seed replication is not embryo replication.

## mmd_u

Problem: Raw MMD is worse than incumbent on all three panels, though better than NLL continuation. Reused panels cannot establish generalization.
Proposed solution: One joint representation/dynamic-regularization configuration using <=8.5 only, with identical joint-representation zero-regularizer control and incumbent reference. Match architecture, initial state, budget and sampling across the new arms. Preserve evaluation space/kernel/calibration so learned representation cannot change the yardstick.
Validation: Prespecify lower mean raw MMD than incumbent and no panel regression under frozen criteria; report paired sampling uncertainty. Fail mean regression or seed-dependent gains. These panels remain exploratory pending independent embryos.

## variogram

Problem: Raw variogram worsens on every panel versus incumbent. Without spatial coordinates this is covariance-sensitive expression structure, not spatial fidelity.
Proposed solution: One scNODE-style jointly trained representation/dynamics regularization trial through8.5, plus identical zero-regularizer control. Match seeds, budget, sampling, full output and evaluation gene pairs; choose parameters on past-only validation.
Validation: Require lower mean raw variogram and improvement on each panel, with no other-metric deterioration under frozen promotion rules. Fail target-informed tuning, gains absent against zero-regularizer, or guardrail violations. Independent embryo validation and readiness remain unmet.

Coordinator decision: no promotion, no more OT grid. Jev selected scNODE protocol/resource preflight (809 input/52 output tokens,1568byte summary). Before any new scientific run review author implementation, lock architecture and matched zero-regularizer control, and verify full-gene decoder adaptation. Current representation limitation is a hypothesis, not proven root cause. Critic suggestions do not relax original success criteria.
