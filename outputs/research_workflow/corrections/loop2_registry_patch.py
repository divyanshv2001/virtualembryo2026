"""Apply C2-6 semantics to the audit snapshot without running legacy stages."""
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parents[1]
root = here.parents[1]
path = here / "hypothesis_registry.json"
before = json.loads(path.read_text(encoding="utf-8"))
protected = ("id", "hypothesis", "rationale", "supporting_evidence", "contradictory_evidence",
             "falsification_criterion", "experiment_required", "current_confidence", "confidence_type")
original = [{k: row[k] for k in protected} for row in before]
ledger = json.loads((here / "evidence_ledger.json").read_text(encoding="utf-8"))
assert {"E003", "E070"} <= {row["id"] for row in ledger}

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

for row in before:
    history = row.setdefault("semantic_correction_history", [])
    if not any(event.get("correction_id") == "C2-6-registry" for event in history):
        history.append({
            "correction_id": "C2-6-registry", "date": "2026-09-28",
            "original_status": row.get("status"),
            "original_empirical_biological_claim": row.get("empirical_biological_claim"),
            "original_status_basis": row.get("status_basis"),
            "note": "Archived scope classification was board-derived, not reviewed scientific scope."
        })
    row["status"] = "untested"
    row["proposal_status"] = "untested"
    row["status_semantics"] = "Proposed intervention or diagnostic remains untested; evidence findings have independent statuses."
    row["empirical_biological_claim"] = None
    row["empirical_biological_scope"] = "unknown/requires_review"
    row["claim_scope_review"] = "Individual scope review required; board is routing metadata and supplies no biological scope verdict."
    row["status_basis"] = "No executed repair or diagnostic experiment adjudicates this proposal as written; observed prerequisite defects are separate evidence."
    row["measured_predictive_evidence_status"] = "untested"
    row["independent_biological_replicates"] = 0

    if row["id"] == "sysbio-r1-1":
        row["critic_disposition"] = {
            "status": "refine", "issue": "C2-6",
            "basis": "Separate observed legacy defect from proposed repair effectiveness; no empirical calibration."
        }
        row["evidence_findings"] = [{
            "finding_id": "sysbio-legacy-verdict-contract",
            "status": "observed code finding",
            "evidence_kind": "source inspection and executed synthetic contract check",
            "claim": "Legacy source maps PLAUSIBLE to supported and UNPROVEN to falsified; a synthetic UNPROVEN fixture produced falsified and repeated ingestion increased history from one to two.",
            "evidence_ledger_ids": ["E070", "E003"],
            "artifacts": [
                {"path": "Neurl IPS 2026/orchestration/memory/update.py",
                 "lines": "34-35,86-103", "sha256": digest(root / "Neurl IPS 2026/orchestration/memory/update.py"),
                 "verification": "source inspected; PLAUSIBLE mapping not independently executed in the recorded fixture"},
                {"path": "outputs/research_workflow/audit_check_results.json",
                 "sha256": digest(here / "audit_check_results.json"),
                 "verification": "recorded synthetic UNPROVEN and repeated-ingestion observation"}
            ],
            "scope": "Legacy code contract; not biological support, falsification or repair validation.",
            "calibration_effect": "none", "independent_biological_replicates": 0
        }]

assert original == [{k: row[k] for k in protected} for row in before]
assert len({row["id"] for row in before}) == len(before)
assert all(row["proposal_status"] == "untested" and row["empirical_biological_claim"] is None for row in before)
path.write_text(json.dumps(before, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
print(f"Corrected {len(before)} audit records; original proposal IDs, claims, evidence and confidence preserved.")
