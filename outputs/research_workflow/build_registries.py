"""Build persistent research ledgers from actual expert artifacts."""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
proposals_dir = ROOT / 'outputs' / 'orchestration' / 'round_1' / 'proposals'
DATE = '2026-09-27'
if any((HERE / name).exists() for name in ('evidence_ledger.json', 'hypothesis_registry.json', 'search_log.json')):
    raise SystemExit('Existing registries preserved. Use explicit amendments; bootstrap must not overwrite adjudications or renumber evidence IDs.')
expert_files = sorted(proposals_dir.glob('*.json'))
ledger = [
    {'id': 'E001', 'claim': 'No biological data, predictions or experiment results are supplied locally.',
     'evidence': 'Complete source inventory; zero h5ad files.', 'source': 'inventory.json',
     'source_type': 'local filesystem observation', 'date': DATE, 'confidence': 'high',
     'supporting_experts': ['orchestrator'], 'contradicting_experts': [], 'status': 'observed'},
    {'id': 'E002', 'claim': 'Existing suite reports 15 passing checks and one missing-board failure.',
     'evidence': 'pytest exit 1: missing PLAN.md board table.', 'source': 'verification.json',
     'source_type': 'executed code check', 'date': DATE, 'confidence': 'high',
     'supporting_experts': ['orchestrator'], 'contradicting_experts': [], 'status': 'observed'},
    {'id': 'E003', 'claim': 'Empty answers become authoritative; UNPROVEN becomes falsified; repeated LEARN duplicates evidence.',
     'evidence': 'Synthetic contract reproductions against unchanged code.', 'source': 'audit_check_results.json',
     'source_type': 'executed synthetic code check', 'date': DATE, 'confidence': 'high',
     'supporting_experts': ['orchestrator', 'sysbio'], 'contradicting_experts': [], 'status': 'observed'},
    {'id': 'E004', 'claim': 'Current official external-data restrictions differ from local blanket T2/T3 prohibition.',
     'evidence': 'Official rules section 10 permits disclosed resources subject to stage/genotype exclusions.',
     'source': 'https://virtualembryo.ai/challenge/rules', 'source_type': 'official rules',
     'date': DATE, 'confidence': 'high for inspected version', 'supporting_experts': ['orchestrator'],
     'contradicting_experts': [], 'status': 'verified public specification'},
    {'id': 'E005', 'claim': 'Embedded project score gains and barcode counts are not verified by this checkout.',
     'evidence': 'Referenced result files and data absent; code assertions are not measured records.',
     'source': 'PROJECT_STATE.md; CODE_AUDIT.md', 'source_type': 'local audit', 'date': DATE,
     'confidence': 'high about missing evidence', 'supporting_experts': ['orchestrator'],
     'contradicting_experts': [], 'status': 'UNVERIFIED historical assertions'},
    {'id': 'E006', 'claim': 'Capture/abundance ambiguity, autonomous reversal and unseen-intervention ambiguity have constructive examples.',
     'evidence': 'Three executed illustrative mathematical checks; negative displacement cosine -0.5 in an autonomous rotation.',
     'source': 'logical_check_results.json', 'source_type': 'executed mathematical counterexample', 'date': DATE,
     'confidence': 'high for logical scope; no biological implication',
     'supporting_experts': ['orchestrator', 'sciml', 'neuralode', 'causal'],
     'contradicting_experts': [], 'status': 'verified logical example; not biological validation'},
]
hypotheses, searches = [], []
for file in expert_files:
    data = json.loads(file.read_text(encoding='utf-8-sig'))
    expert = data['expert']
    searches.append({'expert': expert, 'queries': data.get('searches_run', [])})
    for p in data.get('proposals', []):
        empirical = not (p.get('board') == 'PROCESS')
        hypotheses.append({'id': p['id'], 'hypothesis': p['claim'], 'rationale': p['mechanism'],
            'supporting_evidence': p['evidence'], 'contradictory_evidence': [],
            'falsification_criterion': p['falsifier'], 'experiment_required': p['falsifier'],
            'local_ground_truth': p['local_ground_truth'], 'current_confidence': p['confidence'],
            'confidence_type': 'expert expectation; not empirically calibrated',
            'status': 'untested', 'empirical_biological_claim': empirical,
            'expert': expert, 'board': p['board']})
        for e in p['evidence']:
            ledger.append({'id': f'E{len(ledger)+1:03}', 'claim': e['says'],
                'evidence': e['says'], 'source': e['ref'], 'source_type': e['kind'],
                'verification': e['verified'], 'date': DATE,
                'confidence': 'limited to inspected content; transfer not established',
                'supporting_experts': [expert], 'contradicting_experts': [],
                'status': 'source support as reported by expert; not project-performance evidence',
                'associated_hypothesis': p['id']})
(HERE / 'evidence_ledger.json').write_text(json.dumps(ledger, indent=2), encoding='utf-8')
(HERE / 'hypothesis_registry.json').write_text(json.dumps(hypotheses, indent=2), encoding='utf-8')
(HERE / 'search_log.json').write_text(json.dumps(searches, indent=2), encoding='utf-8')
print(f'{len(expert_files)} expert artifacts; {len(hypotheses)} hypotheses; {len(ledger)} evidence entries')
