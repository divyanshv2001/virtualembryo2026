"""Reproduce orchestration defects using synthetic contracts, not biological data."""
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'Neurl IPS 2026' / 'orchestration'))
import scorer
import rank
import act
from memory import update as learn

class EmptyClient:
    def ask(self, *args, **kwargs):
        return {}

observed = []
entry = scorer.score_proposal(EmptyClient(), {'id': 'empty', 'board': 'PROCESS'}, {'id': 'stats'}, {})
observed.append({'check': 'empty API answers', 'authoritative': entry['authoritative'],
                 'gates': entry['gates'], 'finding': 'Empty answers incorrectly satisfy all(authoritative).'} )
entries = [
    {'proposal_id': 'invalid', 'board': 'PROCESS', 'composite': 1.0,
     'gates': {'violates_data_rules': {'p': 0.99}}},
    {'proposal_id': 'clean', 'board': 'PROCESS', 'composite': 0.8, 'gates': {}},
]
proposals = {e['proposal_id']: {'claim': 'Verify source provenance before ranking scientific proposals',
                              'mechanism': 'Audit provenance and required complete responses'} for e in entries}
rank.dedupe(entries, proposals)
observed.append({'check': 'gated duplicate representative', 'clean_duplicate_of': entries[1].get('duplicate_of'),
                 'invalid_gates': rank.gate_status(entries[0])[0],
                 'finding': 'A clean duplicate is merged into a representative that trips a hard gate.'})

with tempfile.TemporaryDirectory(prefix='contract-audit-', dir=HERE) as scratch:
    out = Path(scratch)
    rd = out / 'round_1'
    (rd / 'proposals').mkdir(parents=True)
    p = {'id': 'stats-r1-1', 'board': 'PROCESS', 'title': 'Synthetic evidence check',
         'claim': 'Synthetic contract only', 'cost': 'free', 'local_ground_truth': 'yes',
         'falsifier': 'Check schema', 'confidence': 0.5}
    (rd / 'proposals' / 'stats.json').write_text(json.dumps({'expert': 'stats', 'proposals': [p]}))
    ranked = {'proposal_id': p['id'], 'expert': 'stats', 'board': 'PROCESS', 'verdict': 'escalate',
              'composite': 0.5, 'axes': {}}
    (rd / 'ranking.json').write_text(json.dumps({'ranked': [ranked]}))
    act.OUT = out
    queue = act.build_queue(1)
    observed.append({'check': 'no critic verdict', 'queued': queue['n_queued'],
                     'status': queue['free_actions_runnable_now'][0]['critic_status'],
                     'finding': 'Missing critic review still enters the action queue.'})
    (rd / 'critic_verdict.json').write_text(json.dumps({'verdicts': [
        {'proposal_id': p['id'], 'expert': 'stats', 'status': 'UNPROVEN', 'reason': 'No data'}]}))
    learn.OUT, learn.MEM = out, out / 'memory'
    learn.update(1)
    first = json.loads((learn.MEM / 'beliefs.json').read_text())['claims'][p['id']]
    learn.update(1)
    second = json.loads((learn.MEM / 'beliefs.json').read_text())['claims'][p['id']]
    observed.append({'check': 'UNPROVEN and missing engine metadata', 'lifecycle': first['lifecycle'],
                     'authoritative': first['history'][0]['judgment_authoritative'],
                     'history_after_first': len(first['history']), 'history_after_repeat': len(second['history']),
                     'finding': 'Unproven becomes falsified; missing engine becomes authoritative; rerun duplicates history.'})

(HERE / 'audit_check_results.json').write_text(json.dumps({'synthetic_only': True, 'observations': observed}, indent=2))
for row in observed:
    print(row['check'] + ': ' + row['finding'])
