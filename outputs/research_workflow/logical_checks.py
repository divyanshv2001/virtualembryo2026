"""Illustrative mathematical counterexamples, never embryonic measurements."""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
capture_a, abundance_a = [1, 1], [50, 100]
capture_b, abundance_b = [.5, 1], [100, 100]
def proportions(c, n):
    weights = [ci * ni for ci, ni in zip(c, n)]
    return [w / sum(weights) for w in weights]
pa, pb = proportions(capture_a, abundance_a), proportions(capture_b, abundance_b)
assert pa == pb

# Autonomous rotation: x'=-omega*y, y'=omega*x. Exact analytic snapshots.
omega = 2 * math.pi / 3
points = [(math.cos(omega * t), math.sin(omega * t)) for t in (0, 1, 2)]
displacements = [[points[i+1][j] - points[i][j] for j in (0, 1)] for i in (0, 1)]
dot = sum(x*y for x, y in zip(*displacements))
norms = [math.sqrt(sum(x*x for x in d)) for d in displacements]
cosine = dot / (norms[0] * norms[1])
assert cosine < 0

# Generic intervention SCMs share WT and observed intervention A; B differs.
worlds = [{'wt': 0, 'do_A': 1, 'do_B': effect} for effect in (-1, 1)]
assert worlds[0]['wt'] == worlds[1]['wt'] and worlds[0]['do_A'] == worlds[1]['do_A']
assert worlds[0]['do_B'] != worlds[1]['do_B']
record = {'scope': 'Executed mathematical checks on arbitrary illustrative values; no biological data or metric validation.',
          'capture_abundance': {'observed_a': pa, 'observed_b': pb, 'observationally_equivalent': True,
                                'conclusion': 'Recovered proportions cannot uniquely identify physical abundance and capture.'},
          'autonomous_rotation': {'points': points, 'displacement_cosine': cosine,
                                  'conclusion': 'Negative consecutive displacement cosine does not falsify all autonomous dynamics.'},
          'unobserved_intervention': {'worlds': worlds,
                                     'conclusion': 'Agreement on WT and intervention A does not identify intervention B without additional assumptions.'}}
(HERE / 'logical_check_results.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print('Three illustrative logical checks passed; displacement cosine:', round(cosine, 3))
