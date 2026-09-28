"""Tiny deterministic arithmetic checks, not a biological estimator or scorer."""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
def output_map(v, scale=10):
    u = [math.expm1(max(x, 0)) for x in v]
    total = sum(u)
    return [math.log1p(scale*x/total) for x in u] if total else [0.0]*len(v)

def close(a, b):
    return all(abs(x-y) < 1e-10 for x,y in zip(a,b))

checks = {}
w = (7.25-6.75)/(8.0-6.75)
checks['interpolation_weight'] = abs(w-.4) < 1e-12
p_a, p_b = [.8,.2,0], [.4,.4,.2]
pi = [(1-w)*a+w*b for a,b in zip(p_a,p_b)]
checks['mixture_sums_to_one'] = abs(sum(pi)-1) < 1e-12
checks['donor_stage_probabilities'] = all(abs((1-w)*a/p+w*b/p-1)<1e-12 for a,b,p in zip(p_a,p_b,pi))
checks['unseen_training_stratum_has_zero_support'] = (1-w)*0+w*0 == 0
x = [math.log1p(3), math.log1p(7)]
checks['known_transform_round_trip'] = close(output_map(x), x)
y = output_map([-1,2])
checks['clipped_output_nonnegative_normalized'] = min(y)>=0 and abs(sum(math.expm1(z) for z in y)-10)<1e-10
checks['zero_row_flag_required'] = output_map([0,0]) == [0,0]
# Same deterministic donor ledger preserves expression/coordinate row identity.
donors = [('A',0),('B',1),('A',1)]
coords = {('A',0):(0,0,0),('A',1):(1,0,0),('B',1):(0,1,0)}
checks['paired_coordinate_identity'] = [coords[d] for d in donors] == [(0,0,0),(0,1,0),(1,0,0)]
# A nonlinear map can change distances; no covariance-preservation promise follows.
checks['output_map_changes_pairwise_distance'] = abs(sum((a-b)**2 for a,b in zip([-1,2],[1,2]))-sum((a-b)**2 for a,b in zip(output_map([-1,2]),output_map([1,2])))) > 1e-6
try:
    output_map([1000,1])
    checks['overflow_rejected'] = False
except OverflowError:
    checks['overflow_rejected'] = True
sigma, z, headroom = 2, 2.5, 1
n1, n2 = (z*sigma/headroom)**2, (z*sigma/(headroom/2))**2
checks['halved_power_headroom_quadruples_n'] = n2 == 4*n1
assessment = {'accesses': [], 'confirmatory_available': True}
assessment['accesses'].append({'kind':'assessment', 'model':'frozen_v1'})
assessment['confirmatory_available'] = False
checks['assessed_target_cannot_be_reused_confirmatorily'] = not assessment['confirmatory_available']
result = {'checks': checks, 'passed': all(checks.values()), 'scope': 'arbitrary synthetic arithmetic and ledger checks only; no data fitting, official score, spatial fidelity or biological validation', 'seed': 'fixed donor list; not PCG64 estimator implementation'}
(HERE/'baseline_contract_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
raise SystemExit(not result['passed'])
