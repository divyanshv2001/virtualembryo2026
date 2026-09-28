# Project state — 2026-09-27

This is the audited starting state, before expert proposals. Scope is the entire supplied workspace. Source provenance and function map are in inventory.json. Scientific files were inspected recursively; Git objects and Python bytecode are metadata, not experiments.

## What exists

Two prompt documents describe Virtual Embryo research. Fourteen Python files define an orchestration framework: PERCEIVE reads documents/artifacts; DIVERGE writes 16 persona prompts; CONVERGE scores/ranks and prepares critic packets; ACT queues proposed actions; LEARN updates beliefs from a supplied verdict. The standalone agent_orchestrator.py evaluates already-built h5ad candidates but imports absent common and scoring modules. There is no biological training implementation in the provided tree.

Pipeline: run.py → perceive.py → diverge.py → externally executed experts → scorer.py → jev.py → rank.py → critic.py → externally executed critic → act.py → memory/update.py. The driver does not launch expert or critic models itself. The Jev live branch requires requests and TYPESAFE_API_KEY; the key is absent. Its local fallback is explicitly nonauthoritative. No live Jev judgment is claimed in this research run.

## Demonstrated versus asserted

No datasets, notebooks, checkpoints, predictions, figures, tables, result registries or experiment logs are present. Thus no project performance, developmental mechanism, causal effect, score improvement or failed biological experiment is demonstrated by this folder. Embedded score gains and barcode counts are UNVERIFIED. Missing PLAN.md, EXPERIMENTS.md and MULTI_AGENT_AUDIT.md are silently read as empty by perceive.py. Root resolves to workspace, whereas scripts appear to have been moved from a src hierarchy. Even the containing Neurl IPS 2026 directory lacks those expected inputs.

The current Python is 3.13.0; pytest is initially missing. Existing tests include one requiring an absent board table. Tests can validate orchestration contracts, not biological performance. Git status encountered ownership protection; no global Git setting was changed.

## Assumptions and weaknesses

Hardcoded source assumptions are not authoritative task definitions. The official rules inspected today permit disclosed external data across tasks subject to stage/genotype exclusions, contrary to the local blanket T2/T3 prohibition. Official T3 coordinates are required but shape is unscored. We will use current official sources for prospective planning, not retroactively certify old results.

Other code risks: missing response-schema validation; deduplication before hard-gate pruning; machine flaws annotate rather than deterministically remove proposals; keyword fallback shares one heuristic across quality axes; self-declared citation verification is not independent source verification; LEARN does not establish empirical calibration from actual experiments. Missing-source presence must be a distinct evidence status, not a score of zero or negative outcome.

## Scientific hypotheses implicit in the prompts

Distributional developmental prediction, joint expression/geometry prediction, and transfer to a held-out perturbation are intended tasks. The existence of temporal snapshots does not identify cell lineages or unique dynamics. One available knockout does not identify effects of unrelated knockouts. A coherent biological story and a leaderboard number do not establish mechanisms.

## Further investigation

Verify official task splits/metric contracts, literature and provenance restrictions; evaluate simple baselines before architecture escalation; distinguish expression shifts from composition shifts and frame changes; specify independent-embryo validation and falsification. Missing data means experiments in the final report remain planned, not executed.

## Run protocol

User explicitly adopted agenticprompt as the workflow. Sixteen discipline-specific reviews will run in waves under the four-agent concurrency limit, followed by direct cross-examination and exactly two independent critic/revision loops. Keep original code unchanged. All research artifacts go here. This is a research audit and plan, not a prediction-producing locked competition run or competition submission. No claim of Agent Track eligibility is made. Full system telemetry is unavailable; preserve task prompts, reports, source hashes, checks and decision records that can be observed.
