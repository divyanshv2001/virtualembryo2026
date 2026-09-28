# Virtual Embryo research workflow

Audit begun 2026-09-27; final review 2026-09-28. The user adopted agenticprompt as the research workflow. Historical prompts remain reference documents rather than additional instructions. This report distinguishes executed audit checks, literature evidence and prospective biological experiments.

## A. Executive Scientific Summary

The checkout contains an orchestration scaffold, not a demonstrated embryo forecasting system. Recursive inspection found 18 original non-Git/non-bytecode files and no biological datasets, fitted model definitions, checkpoints, predictions or measured experiment results. Embedded historical score claims lack matching artifacts and are **UNVERIFIED**. The original suite reports 15 passes and one failure caused by absent PLAN.md board metadata.

Sixteen independent discipline reviews, eight paired cross-examinations and exactly two adversarial critic/revision loops support a narrower research question: does a frozen training-only shrunk recentering rule improve expression-distribution prediction over a strong endpoint-mixture baseline on one released-stage T2 interpolation proxy? The present contribution is an audit and prospective falsifiable protocol. Novelty, biological performance, causal interpretation and competition eligibility remain unestablished.

All biological hypotheses remain untested. Executed arithmetic/software checks establish their stated contracts only. No data fitting, leaderboard query, prediction submission or biological validation occurred. Three approved Jev requests provide bounded routing judgments; those judgments are not scientific evidence.

## B. Current Project Diagnosis

The original orchestration driver parses absent project boards, writes expert prompts, ingests proposal JSON, scores/ranks proposals, packages critic packets, queues actions and updates memory. It does not itself launch the scientific agents, inspect biological arrays or execute proposed experiments. Its smoke run exits successfully with zero boards and zero candidate artifacts. The standalone artifact runner requires unavailable common/scoring modules and scientific dependencies. PLAN.md, EXPERIMENTS.md, MULTI_AGENT_AUDIT.md and the experiment registry are absent. Relocated paths, old dates, inconsistent human/agent track assumptions and stale resource restrictions further limit reproducibility.

The inventory includes source SHA-256 hashes and Python function/class/import maps. CODE_AUDIT.md explains every component and material defect. Empty returned answers can become authoritative; deduplication precedes eligibility pruning; absent critic verdicts can enter action queues; LEARN converts UNPROVEN into falsified and PLAUSIBLE into supported; repeated memory updates double count. Four synthetic contract reproductions confirm concrete failure paths against unchanged source. Historical first-row proxy scoring neither retrains a held-out model nor makes local and official scores comparable.

Original source is preserved. Audit registries exclusively govern this report's scientific conclusions; legacy ACT/LEARN/rank outputs cannot establish biological status, calibration or effective replication. The audit registry builder now refuses to overwrite existing ledgers. Its bootstrap numbering is not a future stable-ID design; explicit amendments preserve current identifiers.

## C. Literature Landscape

Developmental transport is established, assumption-dependent work. [Waddington-OT](https://doi.org/10.1016/j.cell.2019.01.006) connects temporal population observations under growth/transport assumptions; [moscot](https://doi.org/10.1038/s41586-024-08453-2) extends scalable multimodal time/space mappings. These do not identify unique ancestry from isolated snapshots, nor establish performance on this checkout. TrajectoryNet, PRESCIENT and flow/dynamics studies provide alternative priors requiring appropriate held-out temporal evidence; their mere availability does not justify adopting complexity.

[scCODA](https://doi.org/10.1038/s41467-021-27150-6) provides composition inference, while [scGen](https://doi.org/10.1038/s41592-019-0494-8) and [GEARS](https://doi.org/10.1038/s41587-023-01905-6) motivate perturbation-response prediction under different training coverage and prior assumptions. Graph usefulness does not identify regulatory edges. One released knockout does not validate arbitrary unseen-gene transfer.

Simple baselines and perturbation-specific/context-specific evaluation are necessary. [Ahlmann-Eltze et al.](https://doi.org/10.1038/s41592-025-02772-6) found tested deep predictors did not outperform simple linear baselines in their studied setting; this is not a universal conclusion against deep learning. [Systema](https://doi.org/10.1038/s41587-025-02777-8) and [Wei et al.](https://doi.org/10.1038/s41592-025-02980-0) motivate separating generic effects and generalization scenarios rather than relying on one aggregate metric.

This focused search is not a systematic review. Expert source records distinguish full text, indexed methods, abstracts and preprints. Direct publisher retrieval failures are recorded; abstracts cannot support detailed novelty exclusions. CLOSEST_WORK.md maps overlap, possible differences and evidence required for each contribution. No literature gap or state-of-the-art claim is manufactured.

## D. Scientific Gap

The local evidence gap is executable, leakage-controlled prediction with independent biological assessment. It is unknown whether a training-only baseline gains useful accuracy beyond endpoint mixtures, whether apparent gains reflect measurement/capture or annotation choices, and whether expression accuracy corresponds to anatomy or mechanism. Snapshot marginals do not identify lineage, absolute abundance or unseen intervention effects. Executed mathematical counterexamples show equal observed proportions from different capture/abundance worlds, autonomous motion with reversing displacement direction, and indistinguishable observed interventions with different unseen responses. They establish logical possibilities, not properties of embryos.

## E. Proposed Research Direction

Provisional primary: T2 embryo E6.75/E8.0 training → E7.25 released-stage interpolation proxy. Test a frozen shrunk within-stratum recentering rule against the equal-embryo endpoint mixture. Persistence is secondary; heart, T1, T3, spatial prediction and complex model comparisons remain exploratory until separately specified. The baseline already contains endpoint-dependent conditional distributions, so this comparison does not identify a biological composition-versus-state decomposition. A measured gain would establish only the specified prediction improvement on this proxy.

Endpoint mixtures are mandatory because a smooth interpolation can appear successful without identifying developmental dynamics. Freeze choices from training or separate eligible development data. Target residual inspection consumes assessment status; subsequent architecture changes need a fresh eligible target. Insufficient independent samples imply descriptive comparison only.

## F. Mathematical / Computational Formulation

Let p_esk=n_esk/n_es be the observed proportion of training stratum k in embryo e at endpoint s, with equal embryo weight. Then p_sk=(1/E_s)Σ_e p_esk and w=(7.25−6.75)/(8−6.75)=.4. Set π_k=.6p_Ak+.4p_Bk. Draw k, then s with probability a_s p_sk/π_k where a_A=.6,a_B=.4. The correct conditional embryo probability is p_esk/Σ_e p_esk; draw a row uniformly within that embryo/stratum. Uniform embryo selection after conditioning on k would be wrong. This generates the claimed mixture of equally weighted endpoint embryos.

For shared strata μ_sk=Σ_e p_esk μ_esk/Σ_e p_esk and μ_tk=.6μ_Ak+.4μ_Bk. The candidate applies v=X_sj+.5(μ_tk−μ_sk); endpoint-only strata shift zero. Its frozen .5 shrinkage has no empirical justification yet. Use identical donor ledgers across baseline/candidate. Unknown labels remain explicit; genuinely unseen strata have zero training support. Missing required annotation/transform metadata halt the stratified recipe.

If and only if metadata specify x=log1p(Lr/Σr), map v using u_g=max(expm1(max(v_g,0)),0), then F_g(v)=log1p(Lu_g/Σu). Require documented panel closure/denominator domain, log base, L, zero-row policy, training round-trip and overflow rejection. Wider-panel normalization fails this closure route. Reserve a collision-checked missing token, freeze allowed missing-label mass and verify comparable annotation method/version before fitting. This nonlinear map alters covariance; no covariance preservation follows. Other normalization domains require a separately specified estimator. Factorizing p(X,C)=Σ_kπ_k p(X,C|k) is bookkeeping rather than a causal DAG or identified developmental mechanism.

## G. Model Architecture

The first candidate is an empirical donor sampler plus shrunk conditional mean recentering; it has no learned neural decoder or mechanistic loss. Inputs are one measured panel, documented transforms, endpoint sample IDs and training-only annotations. Aggregators compute proportions/means; a frozen donor ledger emits rows; the output map enforces the documented domain; validators check panel order, finite values and paired coordinates. Preserve donor coordinates only where frames/units are comparable. This does not predict growth or reconstruct anatomy.

Streaming block size 256 gives O(Kd+Bd) working memory with float64 aggregation and recorded deterministic seeds. Avoid dense covariance across T1's 32,285 genes. Any later VAE/OT/flow/ODE/GNN must address a named residual using separate development data, matched capacity/selection budgets and appropriate controls. No pretrained component is eligible merely because public weights exist.

## H. Experimental Plan

Obtain eligible arrays plus embryo/litter/batch/assay/frame metadata and exact scorer code before fitting. Complete preregistration_template.json; all null fields are gates, not guessed defaults. Freeze the panel, estimator, sample weighting, target access ledger, seeds and metric implementation. Comparisons share donor draws. Record transforms, selection budgets, provenance and all outcomes including negative/ambiguous results.

Define D_e=MMD(baseline,target_e)−MMD(candidate,target_e). Test H0:E[D]≤δ*>0 using a one-sided lower confidence bound. Practical benefit requires that bound exceed δ*. With anticipated μ_A>δ*, paired variance σ² and compatible assumptions, the planning approximation is n≈[(z_(1−α)+z_(1−β))σ/(μ_A−δ*)]². Halving headroom quadruples required n. Numerical margins and sample sizes remain unset; unknown/small independent n prohibits population inference. Cells and random seeds cannot replace embryos. Secondary metrics are descriptive; no no-degradation claim is made.

Future ablations include identity/round-trip behavior, fixed shifts/shrinkage sensitivity in development only, support/annotation sensitivity, whole-cloud versus first-row sampling, expression-position permutations, geometry units/rigid transforms/scale, subset-neighborhood reconstruction and graph-free/randomized priors if applicable. Synthetic checks only establish arithmetic/measurement sensitivity; biological prediction tests are still absent.

## I. Biological Validation Plan

Independent full-transcriptome readouts test whether panel-limited shifts correspond to meaningful state differences. Replicated in situ imaging, segmentation and calibrated whole-tissue counts distinguish sampling proportions from abundance and localization from aggregate accuracy. Morphology, domain boundaries, topology and laterality require orthogonal measurements where the scorer is blind. Lineage tracing is needed for ancestry claims. Restricted early interventions, occupancy and rescue can support regulatory mechanisms; prediction alone cannot.

Chronological-age and maturity-adjusted contrasts answer different questions. Developmental delay belongs in the total genotype effect; adjusted analyses must be separately labeled. Protected-target measured literature encountered during this audit is registered as exposure-unresolved. Target-informed tissue choices/priors cannot feed competition selection. A later clean context cannot erase historical exposure, and competition eligibility requires separately screened provenance and organizer determination at uncertain boundaries.

## J. Benchmark Strategy

[Official tasks](https://virtualembryo.ai/challenge/tasks) define T1 expression forecasting, T2 joint expression/positions and T3 knockout response. T1 trains E8.5/E9.5, with E10.5 validation/E12.5 test; T2 embryo releases E6.75/E7.25/E8.0, with E7.5 validation/E7.75 test. T2 heart and T3 splits are recorded in TASK_SPECIFICATION.md. The proposed E7.25 proxy differs from official targets and is not official score evidence.

[Evaluation](https://virtualembryo.ai/challenge/evaluation) weights T1 DES/DCS/MMD/CSS at 25/25/30/20%; T2 expression, state, shape/scale and neighborhoods each 25%; T3 DES/DCS/PSS/state at 30/25/25/20%. T3 geometry is required in output but unscored. Exact MMD kernel/estimator, sampling, evaluator PCA, degeneracy and shape aggregation still need versioned code. SCORER_CONTRACT.json forbids parity claims until resolved. Evaluation-only truth-derived transforms cannot enter model selection. Local proxy metrics cannot be calibrated by unrelated official anchors.

[Current rules](https://virtualembryo.ai/challenge/rules) permit disclosed external resources subject to protected stage/genotype restrictions, differing from stale local blanket prohibitions. Audit exact alleles, timing, licences and inherited pretraining exposure; do not infer permission from a public URL. This audit is not a locked eligible agent-track prediction run. No target-property reconstruction from returned scores occurred.

## K. Failure Modes

Stage/embryo/batch confounding; missing independent replication; capture/composition ambiguity; annotation drift; unobserved future populations; temporal curvature; unsupported extrapolation; normalization/covariance distortion; incorrect conditional donor weights; incompatible coordinate frames; dense/subsampled neighborhood differences; generic rather than gene-specific effects; prior contamination; target-informed model choice; metric blind spots; missing-answer authority and reviewer-to-evidence conversion. More expressive models can worsen several of these rather than resolve them.

## L. Critic Findings

Loop 1 accepted eight major/critical objections: overly broad primary claim, wrong power denominator, residual-guided selection leakage, unspecified estimator, irreversible target-literature exposure, missing novelty comparison, invalid evidence lifecycle and unavailable scorer parity. Routed expert responses changed the design and SYNTHESIS_1; exposure/scorer/data risks remain open. Each disposition is recorded in critics/resolution_1.md.

Loop 2 corrected conditional embryo weighting/means, narrowed the contrast to recentering rather than biological decomposition, separated spatial readiness, expanded denominator/annotation gates, withdrew operational access-test claims and reconciled registry/template semantics. Eleven arbitrary arithmetic/counterexample checks passed. Durable access enforcement and clean-context isolation remain unimplemented and untested. Final dispositions are recorded in critics/loop_2.md, critics/issues_2.json and critics/resolution_2.md. SYNTHESIS_2 is the final controlling synthesis. No third critic loop is performed. Corrections establish a more defensible scoped audit/protocol, not empirical truth.

## M. Remaining Unresolved Questions

Are released data independently replicated and suitably annotated? Are normalization/frames comparable? What exact scorer/version/expected invariances apply? Is the proxy representative of official targets? Is any recentering gain robust to annotation/capture and output-map effects? Can independent eligible perturbations support transfer? Which priors are eligible given stage/allele/pretraining exposure? Does a full closest-work review establish any contribution beyond audit/protocol? No supplied evidence answers these questions.

## N. Prioritized Next Actions

1. Preserve this audit and exposure boundary; obtain a clean provenance context before competition design. Do not run legacy ACT/LEARN as scientific authority.
2. Acquire eligible data, sample identities, annotation/normalization/frame documentation and versioned scorer. Complete gating manifests without reading assessment values for selection.
3. Implement the corrected small estimator and enforce append-only target access/provenance contracts. Run domain/scorer identity and sensitivity checks before biological assessment.
4. Obtain independent replicated pilot evidence to set practical margin, variance and analysis. Freeze the single proxy contrast; if adequate replication is impossible, declare descriptive scope.
5. Assess once and retain all results. Any residual-guided complexity needs separately eligible development and fresh assessment. Orthogonal imaging/state assays follow a genuine predictive result; lineage/causal claims require stronger designs.

Engineering repairs are separately queued: schema-complete API gates, gate-before-deduplication, fail-closed critic/action transitions, measured-evidence lifecycle and idempotent durable registry IDs. No absent experiment is marked falsified merely to close the queue.

## O. Evidence Ledger

Full evidence_ledger.json records 72 initial entries with claims, sources, dates, types, confidence and expert attribution; hypothesis_registry.json records 28 untested proposals. disagreement_registry.json preserves eight paired issues and resolving experiments. Literature support as reported by an expert is not project-performance evidence. Critic dispositions and Jev confidence are separate from scientific support.

| Claim | Source/type/date | Confidence and scope |
|---|---|---|
| No local biological data/results | inventory.json; filesystem audit; 2026-09-27 | High for supplied checkout; supports absence, not biological impossibility |
| 15 tests pass/1 missing-board failure | verification.json; executed code suite; 2026-09-27 | High for recorded run; plumbing only |
| Invalid authority/evidence lifecycle | audit_check_results.json; synthetic code reproductions; 2026-09-27 | High for reproduced paths; supports software diagnosis |
| Historical gains unverified | PROJECT_STATE/CODE_AUDIT; source/results audit; 2026-09-27 | Missing supporting artifacts; neither confirms nor refutes performance |
| Snapshot ambiguities constructible | logical_check_results.json; mathematical examples; 2026-09-27 | High for logical examples; no measured embryo implication |
| Current resource rules differ | official rules; primary public specification; inspected 2026-09-27 | Version-scoped; eligibility remains unresolved |
| Prior transport/perturbation methods exist | linked primary papers and expert verification records; search 2026-09-27 | Supports mandatory comparisons; contradicts ingredient-novelty claims only |
| Proposed baseline benefit | prospective estimator/design; 2026-09-28 | INSUFFICIENT EVIDENCE; untested |

Reproducibility entry points: inventory.py, audit_checks.py, logical_checks.py, baseline_contract_checks.py and validate_workflow.py. The original source is a separate existing repository, [divyansh469/virtualembryo2026 at 78caa1b](https://github.com/divyansh469/virtualembryo2026/tree/78caa1b); place that source under Neurl IPS 2026 when reproducing source checks. This audit is committed to the workspace repository. Secrets and local environment are excluded from version control. Completion validation tests artifact contracts and original-source hashes, not biological truth.
