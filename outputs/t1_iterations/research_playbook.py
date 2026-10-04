"""Compact, disk-backed resume entry point; no model calls or large data loads."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACKED = ('LOCAL_OPTIMIZATION_STATE.json', 'LOCAL_OPTIMIZATION_PROMPT.md',
           'METRIC_RESEARCH_QUEUE.json', 'METRIC_CRITIQUE_POLICY.json',
           'METRIC_CRITIQUES_20261001.md', 'RESEARCH_HARNESS_MANIFEST.json')


def fingerprints():
    return {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
            if (HERE/name).exists() else None for name in TRACKED}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    manifest = HERE/'PLAYBOOK_STATE.json'
    current = fingerprints()
    previous = json.loads(manifest.read_text()) if manifest.exists() else {}
    changed = [name for name in TRACKED
               if current[name] != previous.get('fingerprints', {}).get(name)]
    if args.refresh:
        state = json.loads((HERE/'LOCAL_OPTIMIZATION_STATE.json').read_text())
        compact = {'updated_utc': datetime.now(timezone.utc).isoformat(),
                   'fingerprints': current,
                   'objective_status': state['objective_status'],
                   'process_state_is_snapshot': True,
                   'active_jobs_at_checkpoint': state.get('active_jobs', []),
                   'next_experiment': state['next_experiment'],
                   'export': state.get('formula_progress_export'),
                   'reward': state.get('metric_critique_reward'),
                   'official_best': state.get('best_reported_official_score'),
                   'latest_official': state.get('latest_official_result'),
                   'latest_jev': state.get('latest_jev_advisory'),
                   'external_qc_audit': state.get('external_qc_audit')}
        manifest.write_text(json.dumps(compact, indent=2)+'\n')
        export = compact['export'] or {}
        official = compact['latest_official'] or {}
        playbook = f'''# T1 resume playbook

Updated: {compact['updated_utc']}

Run `outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/research_playbook.py` first. Its machine-only hashes check policy/state changes without sending whole documents into model context. Read this short playbook and only the changed policy sections or active experiment evidence needed for the next action. Never replay all Markdown/history by default.

Use `research_harness.py queue` for a compact next-step/pending summary; never dump METRIC_RESEARCH_QUEUE.json. Use `research_harness.py status --experiment NAME` for the current experiment; omit NAME only for a needed global audit. This gives compact live-process/report checks, `run --experiment NAME` for registered launches and `collect --experiment NAME` for cached critic packets. Commands use the same project Python runtime. Register a new predeclared script/run/report before launch. Collection prepares packets; the coordinator still runs four critics and Jev decisions. See RESEARCH_HARNESS.md only when changing orchestration. Official task review: TEMPORAL_TASK_REVIEW_20261001.md.

Official best: {compact['official_best']}. Latest reported skills: {official.get('metric_skills_percent', 'see original official result')}. Covariance .25 reused-development mean55.98883; EB and lineage-standardized historical proxy screens failed. Local>72 mean/lower-tail >=64-replicate AND temporal gate unmet. Full32285-gene scorer/calibration unchanged.

Checkpoint jobs: {compact['active_jobs_at_checkpoint']}. This snapshot is not a live process check: inspect actual Python processes and active report/events before launching; never duplicate work.

Next: {compact['next_experiment']}

Completed export: {export.get('artifact', 'none')}. Format passed; SHA256 {export.get('artifact_sha256', 'none')}.1500cells/32285genes,30.55MB. Latest user-reported official result: {official.get('headline_score', 'none')}; attribution/screenshot evidence in {official.get('report', 'export report')}. No automatic upload. Source throughE9.5 and observedE9.5; stale generic audit wording annotated.

Four metric critiques completed; consult METRIC_CRITIQUES_20261001.md when designing the next mechanism. Separate reward starts0: +1 positive paired mean skill gain / -10 tie or regression per metric, aggregate maximum100; negative balances visible. Pending/invalid/proxy0. Future unique declared experiments only; promotion gates unchanged.

Current separate reward: {(compact['reward'] or {}).get('current_reward', 0)}. This is not a benchmark score.

External acquisition entry point: {(compact['external_qc_audit'] or {}).get('strict_manifest', 'original metadata audit')}. The old stage-only list is historical;127 mutants excluded,162 B6 WT controls separate,143E8.5/999E9.5 primary CD1 candidates. Expression acquired: {(compact['external_qc_audit'] or {}).get('expression_acquired', False)}; acquired cohort cells: {(compact['external_qc_audit'] or {}).get('acquired_cohort_cells', 0)}; prospective QC passes: {(compact['external_qc_audit'] or {}).get('prospective_qc_passing_cells', 0)}; original pilot cells: {(compact['external_qc_audit'] or {}).get('pilot_cells', 0)}; training readiness: {(compact['external_qc_audit'] or {}).get('training_approved', False)}. Latest external status: {(compact['external_qc_audit'] or {}).get('compact_status', 'Inspect the latest listed cohort/compatibility reports before declaring new work.')}. Latest Jev actual usage: {(compact['latest_jev'] or {}).get('usage', 'none')}.

Use D-only temporary disk cache, two BLAS threads, one full forecast at a time;16GBRAM. Retain source reports/events, seeds/splits/hashes, raw4/skills4 and failed outcomes. Refresh SCORE_LEDGER.jsonl after full-panel batches; proxy/export is not a score batch. Commit small evidence/code, never datasets/secrets/artifact matrices. No automatic official upload. Run four compact metric critics after each new experiment batch once; max150words each, fresh context, no routine agents. Each metric specialist must state Problem / Proposed solution / Validation, with evidence, uncertainty, concrete past-only remedy, matched controls and failure criteria; proposed solutions are unvalidated until tested.

Jev: cached advisory <=3000bytes for a genuinely new uncertain decision; at most one per decision. Prefer Jev for distinct consequential selection/design/diagnosis/critique-synthesis decisions; multiple decisions per batch allowed. Routine checks use zero calls. Actual last formula calls815in/71out and730in/69out; no measured Codex credit savings. Do not invent quota resets.

After meaningful changes: update state, queue and append concise RESEARCH_UPDATES.md entry; run this tool with `--refresh`. Histories are retained as evidence, not repeatedly loaded. A changed hash is a pointer to inspect relevant changes, not an instruction to read every file. A missing required input is recorded precisely. Never treat the playbook as proof a process is still running.
'''
        (HERE/'PLAYBOOK.md').write_text(playbook)
        print(json.dumps({'status': 'refreshed', 'playbook': 'PLAYBOOK.md',
                          'next_experiment': compact['next_experiment']}))
    else:
        print(json.dumps({'status': 'needs_refresh' if changed else 'unchanged',
                          'changed_sources': changed, 'entry_point': 'PLAYBOOK.md',
                          'active_jobs_at_checkpoint': previous.get('active_jobs_at_checkpoint'),
                          'live_process_check_required': True,
                          'next_experiment': previous.get('next_experiment'),
                          'history_replay_required': False}))


if __name__ == '__main__':
    main()
