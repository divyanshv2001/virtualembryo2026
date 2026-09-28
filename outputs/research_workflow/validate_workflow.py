"""Validate research artifacts without treating plans as biological results."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
IDS = 'devbio scgenomics spatialtx sysbio grn causal genmodel diffusion neuralode gnn spatial3d replearn stats sciml biovalid benchdesign'.split()
errors = []
for expert in IDS:
    review = HERE / 'experts' / f'{expert}.md'
    proposal = ROOT / 'outputs/orchestration/round_1/proposals' / f'{expert}.json'
    if not review.exists() or not proposal.exists():
        errors.append(f'Missing initial expert artifact: {expert}')
        continue
    data = json.loads(proposal.read_text(encoding='utf-8-sig'))
    if data['expert'] != expert or data['round'] != 1:
        errors.append(f'Identity/round mismatch: {expert}')
    for p in data['proposals']:
        for field, cap in {'title': 90, 'claim': 320, 'mechanism': 480, 'falsifier': 280}.items():
            if len(p[field]) > cap:
                errors.append(f'{p["id"]} exceeds {field} cap')
        if not p['evidence'] or not p['falsifier']:
            errors.append(f'{p["id"]} lacks evidence/falsifier')
for row in json.loads((HERE / 'inventory.json').read_text())['source_files']:
    path = ROOT / row['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
        errors.append('Original source changed: ' + row['path'])
for loop in (1, 2):
    for name in (f'critics/loop_{loop}.md', f'critics/issues_{loop}.json',
                 f'SYNTHESIS_{loop}.md', f'critics/resolution_{loop}.md'):
        if not (HERE / name).exists():
            errors.append('Missing loop artifact: ' + name)
for expert in IDS:
    if not (HERE / 'debate' / f'{expert}.md').exists():
        errors.append('Missing direct cross-examination artifact: ' + expert)
for name in ('REPORT.md', 'evidence_ledger.json', 'hypothesis_registry.json',
             'disagreement_registry.json', 'research_queue.json', 'run_manifest.json'):
    if not (HERE / name).exists():
        errors.append('Missing final artifact: ' + name)
report = {'artifact_errors': errors, 'original_sources_unchanged': not any('source changed' in e for e in errors),
          'expert_count_required': 16, 'critic_loops_required': 2,
          'note': 'This validates artifact completeness and contracts, not empirical scientific truth.'}
(HERE / 'completion_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
raise SystemExit(bool(errors))
