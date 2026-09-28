# Recursive code audit

All 18 original non-Git/non-bytecode files were inventoried (including .DS_Store as opaque metadata). Three independent reconnaissance reviewers read the complete code groups; orchestrator inspected perceive/run, contracts and critical branches. JSON inventory maps every Python function/class/import with source hashes. No scientific data or historical results are present.

## Architecture and implementation

| File | Actual behavior | Dependencies or missing evidence |
|---|---|---|
| agenticprompt | Adopted research workflow, 16 experts + critic + exactly two revisions | Not executable code |
| prompts.md | Two historical prompts, including conflicting agent/human track framings | Reference material, not extra task requirements |
| agent_orchestrator.py | Discovers/validates existing h5ad candidates, optional diagnostic proxy, filename-based selection | numpy, anndata, absent common/scoring modules, absent artifacts |
| orchestration/run.py | Drives disk-based stages; waits for experts/verdicts | Sibling modules; cannot spawn generative agents |
| perceive.py | Parses PLAN/EXPERIMENTS/AUDIT and artifact glob; absent inputs silently empty | No raw biological inspection or recursive source inventory |
| personas.py | 16 persona records and peer graph | Historical mandates reference absent src/data; premises UNVERIFIED |
| diverge.py | Writes task prompts/contracts; builds peer digest | Hardcoded 2026-09-19 and human-track wording |
| jev.py | Content-hash cache, live requests, explicit keyword fallback | requests only on live branch, TYPESAFE_API_KEY absent |
| scorer.py | 11 rubric questions per proposal, clipped evidence/state | Incomplete required-answer validation |
| rank.py | Composite, hard gates, lexical Jaccard deduplication, escalation list | Nonauthoritative rankings are heuristic |
| critic.py | Schema/keyword checks, escalation packet, verdict ingestion | Does not launch critic or verify paper content |
| act.py | Serializes proposed actions and checklist | Does not execute experiments |
| memory/update.py | Updates beliefs/calibration/weights from verdict | Critic opinion is not experimental evidence |
| test_orchestration.py | 16 plumbing tests | pytest; one requires missing PLAN.md |
| Two __init__.py files | Package docstrings | No logic |
| .gitattributes / .DS_Store | Line-ending configuration / macOS metadata | Not scientific results |

## Material defects and evidence

1. Missing-answer gates: jev.py:149 trusts returned keys; scorer.py:209–229 can omit gates and mark an empty set authoritative via all(); rank.py:66/79 skips missing values. A schema-complete response is needed before any live result is usable.
2. rank.py:99–142 deduplicates by score before pruning. A disqualified representative can hide a clean near-duplicate. Gate admissibility must precede representative selection.
3. critic.py:251 computes machine flaws after ranking; :267–284 still sends escalated proposals with blocking flaws. Deterministic defects annotate rather than automatically prevent promotion.
4. act.py:80 excludes only REJECTED; missing/UNREVIEWED/UNPROVEN verdicts may enter an action queue. Queue text does not establish admissibility.
5. memory/update.py:34/86 maps UNPROVEN to falsified and PLAUSIBLE to supported. That confuses absence of evidence with refutation and reviewer plausibility with measured evidence. :68 treats absent engine metadata as authoritative. :91/120 permits repeated-round double counting.
6. jev.py:202–221 uses one keyword heuristic for all four-level axes, with shared context contamination. :127–160 counts successful responses rather than attempts and repeats failures for later proposals. scorer.py round labels derive client.live, not the engines of actual answers.
7. critic.py:178–190 trusts declared citation verification/year/path existence. This does not establish that a paper supports a claim. :165–216 keyword checks can flag explicit rejections of a trap.
8. agent_orchestrator.py:175 samples first <=150 rows, does not regenerate a proxy-specific method; :247 compares a proxy score against a real-board baseline. Filename matches at :225 are not provenance checks. Repeated rounds at :271 repeat the same candidates rather than critic revisions.
9. act.py:29 quota is hardcoded per board/day whereas current official rules describe development limits per task/day. Its external-data checklist is stale. Prospective submission tooling requires source versioning.
10. test_orchestration.py:49 checks incoming peer coverage, not connectivity; :210 snapshots an unrelated client's usage before invoking critic; :105 writes normal round-1 prompt paths. No ACT/memory or biological validity coverage.

These are scientific audit findings, not repairs silently applied to the original framework. Source remains unchanged. Existing-suite results and smoke run are in verification.json. The research report and persistent registries are authoritative for this audit; generated action queues and LEARN statuses retain their original, limited semantics.
