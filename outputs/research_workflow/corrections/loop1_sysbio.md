# Critic loop 1 correction — C1-7

I accept C1-7. A prose warning does not prevent downstream consumption of legacy belief or action files. The following boundary applies to this research audit and must be copied into the final report and future repair queue. No original source is repaired by this correction.

## Exclusive scientific decision surface

Within this audit, only `outputs/research_workflow/hypothesis_registry.json`, `evidence_ledger.json`, `disagreement_registry.json`, and the linked primary check/experiment artifacts may establish the audit's scientific status. These registries are authoritative for bookkeeping, not automatically authoritative about biology: every status still needs a scoped source. The final report is a rendering of these registries and their evidence. Critic issues, reviews and routing outputs are disposition records.

Legacy `outputs/orchestration` ACT queues, LEARN beliefs/calibration/strategy and rank products cannot establish biological support, falsification, effective biological sample size or empirical confidence calibration. No proposed action is executed merely because it appears in an ACT queue. This boundary remains necessary while original source is unchanged and is a prerequisite for any later execution plan.

Use distinct fields, not a single overloaded status:

| Field | Allowed interpretation |
|---|---|
| `hypothesis_status: untested` | No measured predictive comparison that adjudicates the stated hypothesis exists. Missing data and UNPROVEN remain here. |
| `hypothesis_status: supported by measured predictive evidence` | An executed, admissible comparison supports the prespecified prediction claim within its stated task/horizon/replicate scope. This does not establish a biological mechanism. |
| `hypothesis_status: contradicted by measured predictive evidence` | An executed admissible comparison contradicts that claim under its prespecified decision rule. Missing evidence and failure to reach a decision threshold are insufficient. |
| `hypothesis_status: logically nonidentifiable under assumptions` | A constructive equivalence or formal argument shows the quantity is not identified from the stated observations/assumptions. This is a logical result, not measured predictive failure. |
| `critic_disposition` | Retain/refine/withdraw/reject/needs-evidence, with issue ID and rationale. Never derives biological status by itself. |
| `evidence_kind` | Filesystem observation, observed code behavior, mathematical example, public specification, literature evidence, or measured project prediction. |

All biological forecast/compensation claims in this supplied workspace remain untested. The capture/abundance example in `logical_check_results.json` supports nonidentifiability of physical abundance from recovered proportions under unknown capture; it does not refute an observed-composition predictor. `audit_check_results.json` supports observed code behavior in synthetic contracts, not biological support. Independent biological n remains zero regardless of expert agreement or number of critic reviews.

## Audit of the current registry builder

I inspected `outputs/research_workflow/build_registries.py`. Lines 15,19,23 correctly call filesystem/code-check evidence “observed.” Lines 36–39 correctly limit mathematical checks to logical examples rather than biology. Lines 52–53 label all proposals “proposed; not tested” and expectation as uncalibrated; safe in intent, but the hypothesis label should be normalized to `untested` with a separate proposal disposition. Lines 59–61 correctly label expert-reported source support as distinct from project performance.

Two implementation limits need a future repair entry. Line 47 infers biological scope solely from `board != PROCESS`; PROCESS diagnostics can discuss biological attribution, while non-PROCESS proposals may be pure format checks. Replace that inference with explicit, reviewed claim scope and evidence kind. Lines 56–63 assign evidence IDs by current list length and overwrite output files; a fixed input is a snapshot rather than an append, but inserting/reordering a proposal can change IDs and rerunning can erase later adjudications. Stable content-linked IDs and preservation of independently stored status/disposition records are needed. I did not execute this builder or modify its outputs in this correction.

## Durable future repair and validation queue

1. Replace legacy PLAUSIBLE→supported and UNPROVEN→falsified mappings (`Neurl IPS 2026/orchestration/memory/update.py:34–35,86–103`) with separate critic/evidence schemas before scientific reuse.
2. Require complete validated response/engine provenance; unknown metadata cannot silently confer authority. Keep code-verification authority separate from biological evidence.
3. Key review events by claim/version, evidence artifact hash, comparison and event ID. Repeated ingestion must not append duplicates, update calibration again, or change biological n/confidence. A changed evidence artifact is a new event, not a repeat review.
4. Gate invalid candidates before deduplication and prevent unreviewed actions from execution. These are execution contracts, not scientific falsifiers.
5. Preserve registry adjudications during rebuild; create stable evidence IDs and include linked evidence hashes, scope, measurement units, independent replicate count and decision rule.

Required isolated code checks: UNPROVEN without experiment evidence preserves untested; PLAUSIBLE without evidence preserves untested; an observed schema bug remains observed code evidence; a constructive capture equivalence is logical evidence only. Reingest each identical event twice and require identical scientific registry, calibration inputs and biological n. Rebuild unchanged source inputs twice and require identical semantic registries; insert an unrelated proposal and require existing IDs/adjudications to survive. These checks are planned repair acceptance tests. The existing synthetic legacy test observed history lengths 1 then 2, documenting the defect rather than passing idempotence.

Disposition: C1-7 is addressed by this explicit audit boundary and repair specification once adopted in the final report/registries. Original legacy behavior remains defective; no source repair or biological validation is claimed.
