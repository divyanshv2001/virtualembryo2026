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

Observed population preflight: shared-name decomposition verified (max error 9.992007221626409e-16); L2 norms {"observed_shift": 5.588031694648865, "shared_within": 3.400593695528512, "shared_composition": 8.413485064255092, "unmatched": 8.571031277166087}. Terms cancel; these are neither contribution percentages nor causal evidence. Stage labels alone do not identify transitions; minimum shared-type support=47. Prior KMeans state slopes and marker donor weighting already tested. A new transition model requires a frozen common state space and independently assessed temporal transfer. No score/training/reward change.

## Deadline action plan — T1 through 8 October 2026 (IST)

Target first place; the supplied Agent Team screenshot leads at68.4 versus our user-reported52.16, a16.24-point gap. Live leader verification was unavailable, so68.4 is a screenshot reference, not a current confirmed threshold. Neither new algorithms nor more GPU time guarantee first place. Keep52.16 as the fallback; do not export a regression as an upgrade.

| Milestone | Concrete work | Decision/output |
|---|---|---|
| Next active session / 5 October | Finish the common-state support decision within two working hours. Use permitted existing atlas metadata and both real observed matrices; verify marker agreement rather than infer lineage from stage names. Inventory only directly relevant previous population methods. | Freeze one genuinely distinct, implementable hypothesis with training cutoff, source hashes, exact controls and failure criteria. If support fails, record why and choose the fullgene supervised-decoder ablation; no further open-ended preflight. |
| 5 October | Reproduce the existing export pipeline and gene order; quantify real-stage training participation and 5510 copied genes. Reuse cached normalization/features/checkpoints. Predeclare one new state-conditional transition/emergence mechanism, with fullgene supervision an explicitly separated ablation. | One registered RTX3060 pilot, not another architecture survey or parameter grid. Tiny finite-gradient/checkpoint smoke first; then full-panel predictions frozen before scoring. |
| 6 October | Run matched past-only copy/incumbent/new-method comparisons on the declared horizons; score all32285genes and allfourmetrics. | Screen with3 matched scoring seeds. Continue only if aggregate improves and differences exceed paired resampling uncertainty; ties/invalid calibration/repeated regressions stop that mechanism. Positive partial gains remain evidence, but do not establish readiness. |
| 7 October | Allocate compute to the surviving mechanism; test another training seed and another permissible chronological horizon before expensive replication. Freeze selection before confirmation. | Confirm allfour rawmetrics/skills, failure cases, resource use and temporal transfer. Run the existing64-replicate mean/lower-tail and temporal readiness gates only for a viable survivor; repeated scoring seeds are not independent embryos. |
| 8 October | Freeze best eligible result, produce/validate one submission artifact and genuine producing-run provenance; retain current verified artifact if no upgrade passes. | Hand off T1 artifacts and concise evidence so effort can move to T2/T3. No automatic official upload. |

Stop broad literature searching, failed potential/MMD objective repeats, old shift/covariance/latent/KMeans/marker-resampling grids, additional external acquisition and routine status agents. Existing52.16 remains the benchmark until a new official result is reported. Prior local55.99 is development evidence, not an official forecast. MMD/CSS jointly weigh50%, so population fidelity and co-expression are priority hypotheses; DE/direction cannot be sacrificed without measuring the weighted tradeoff.

Execution: one training worker, RTX3060 <=4.5GiB allocated VRAM,16GB host cap, D-only cache and unchanged scorer/calibration. Harness launches/collects; Jev provides cached bounded decisions; four compact independent metric reviews follow each new batch. Reuse files and frozen controls; checkpoint meaningful milestones only. Time-box implementation/preflight, not scientific correctness; do not bypass gates or claim human-guided development satisfies locked Agent Team evidence. Readiness >72 mean/lower-tail and temporal gates remain unchanged. An eligible Agent Team artifact requires a separately frozen autonomous producing run; budget this before packaging.

Cached Jev deadline choice: validation_first, confidence.72,1437input/43output tokens. Coordinator decision: keep validation-first safeguards and fallback, but retain one time-boxed new-mechanism pilot because the user explicitly prioritizes rapid score improvement. Jev saw bounded options/report status, not the full numeric audit; this is advice, not certification.

2026-10-04T18:40:08.603379+00:00 Fullgene decoder complete: neural41.21755/linear53.10552/incumbent55.98883;12validcalibs,4freshreviews,primaryreward-40/health-1684. LinearDE/DCS/MMDpartial gains retained, CSSregresses. Exactforecastdensity replay:10633/12303newpositivegenes/cell from3891donorpositive; nocausalclaim. RTX194.3MB/host5.735GB. Linear fitsall16787cells; neural400x128minibatches saw15987uniquerows of16787eligible. CachedJev1089in54out selectsfullgenehurdle noveltypreflight(.11); nooldgates/grids repeat,no newexport/upload.

2026-10-05T01:33:49.501976+00:00 Hurdlepilot56.68901 vs55.98883 (+.70018),all4meanincumbentgains/3aggregatewins,12validcalibs/fourfreshreviews. RTX161.8MB/host6.125GB,guardbackoff1/.38665. Primaryvsdense reward-18/health-1702. CachedJev1141in57out trainingstability(.27); fixeddisjointhalves confirmation declared,notthirdhypothesis. Readiness/60expansiongates unchanged,official52.16/noexport/upload.

2026-10-05T02:01:27.555308+00:00 Fixedhurdlestability:full56.689/half0 56.590/half1 56.707 vs55.989;15validcalibs,exactcontrols,4reviews. Aggregatebeatsinc3/3bothhalves, buthalf0DCS-.000654588failsboth-halves-all4gate; no half1posthocselection/promotion/independentclaim. Fixedconfirmationreward0/health-1702. Onlyreal8.5/9.5fullgene files,celltype-onlymetadata verified; no additionaltemporal/embryo support. CachedJev1280in50out evidencepreflight(.20). Preserve56.69pilot/official52.16fallback; gates/noofficialuploadsunchanged.

## Frozen-run evidence preflight — 5 October 2026

Bounded source audit complete; inspected source hashes are retained in LOCAL_OPTIMIZATION_STATE.json. No live Python workers at resume; no new model, score batch, critic, advisory or reward.

The original outputs/t1_run/run_t1.py reads both real stages and implements seeded persistence plus optional exact-label celltype mean shifts (E9.5 minus E8.5), clipping negatives and leaving unmatched E9.5 types unchanged. Its cached report confirms format/round-trip checks, 11 shared and 10 last-only types, not forecast improvement. These are public-style reference comparators; exact official starter/baked-floor parity remains unverified. baseline_contract_results.json proves only synthetic arithmetic/ledger checks.

The best formula exporter reuses the external through-E9.5 encoder/CNF checkpoint from export_anchor_slope_submission.py and fits real E9.5 anchor slopes/alignment. It does not jointly supervise the temporal field with real E8.5/E9.5; 5510 unsupported genes retain donor values. Both real files have full32285 genes and celltype-only metadata; no embryo IDs or extra real stage were verified. The recent fullgene hurdle fits real E8.5 and assesses previously exposed real E9.5, which is development evidence. The existing runner freezes copy/incumbent/fullfit predictions before scoring and checks archived hashes; this does not establish an independent biological control.

Formula plan/report/trajectory agree on the genuine export-plan hash e19d93a820e7479d418a5bf8e9b1b8d1c747e87880c97c6b97b2a994644eaae0. That freeze covers the export recipe, not a pre-run autonomous Agent Team configuration. outputs/agent_evidence/README.md expressly records no configuration lock in current interactive development; exported archives and trajectories cover their actual scope only. Do not reconstruct a lock or substitute this harness for older producing runs.

Both permitted deadline hypotheses have been tested. Fullfit56.68901 versus incumbent55.98883 remains positive development evidence; fixed halves56.59017/56.70664 retain aggregate gains but half0 direction-.000654588 fails the declared all-four stability criterion. No half1 posthoc choice, promotion, >=60 expansion, or original >72/64-replicate/temporal readiness claim. No third broad architecture/grid follows this audit. Next preserve official52.16 fallback and existing truthful evidence through the deadline; a future eligible Agent Team prediction requires a genuinely prospective locked producing run and the unchanged scientific gates. No official upload.

2026-10-05T02:42:43.427754+00:00 Fixedridge2 stabilization failed: full56.49849 vsoriginal56.68901, halves56.47834/56.45206 vsinc55.98883. Directionmean half0 now+0.001682330, butMMD allnewarms belowinc and fullmeanretention fails.18validcalibs/4freshreviews; oldfailure/pilotretained,no promotion. RTX161.8MB/host6.132GB; primaryreward-18/health-1720.
