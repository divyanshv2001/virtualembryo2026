"""Separate research reward; never changes the benchmark scorer.

Each evaluated metric earns +1 for a strict paired mean skill gain and -10
otherwise. Four metric balances sum to a reward capped at 100. Start at zero;
negative balances remain visible. Pending/invalid experiments earn no reward.
Only future, predeclared full-panel experiments qualify; proxies do not.
"""
import math

METRICS = ('de_score', 'de_direction', 'mmd_u', 'variogram')


def assess(candidate_skills, incumbent_skills, *, eligible=True):
    """Consume matched full-panel replicate skills in canonical [0,1] units."""
    if not eligible:
        return {'status': 'ineligible', 'reward_delta': 0, 'metrics': {}}
    if not candidate_skills or len(candidate_skills) != len(incumbent_skills):
        raise ValueError('Nonempty matched replicate sets required')
    result = {}
    for metric in METRICS:
        differences = []
        for candidate, incumbent in zip(candidate_skills, incumbent_skills):
            a, b = float(candidate[metric]), float(incumbent[metric])
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in (a, b)):
                raise ValueError('Expected finite canonical skills in [0,1]')
            differences.append(a-b)
        gain = math.fsum(differences)/len(differences)
        result[metric] = {'paired_mean_skill_gain': gain,
                          'positive_pairs': sum(v > 0 for v in differences),
                          'pairs': len(differences),
                          'reward': 1 if gain > 0 else -10,
                          'stability_verified': False}
    return {'status': 'evaluated', 'metrics': result,
            'reward_delta': sum(v['reward'] for v in result.values())}


def total_reward(events):
    ids = [event['experiment_id'] for event in events]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate experiments cannot earn repeat rewards')
    uncapped = sum(event['assessment']['reward_delta'] for event in events)
    return {'uncapped_reward': uncapped, 'reward_score': min(100, uncapped),
            'maximum': 100, 'benchmark_score': 'unchanged; separately reported'}


def health_snapshot(events, redline=0):
    """Derived health indicator; immutable reward events remain authoritative."""
    total=total_reward(events)
    balances={metric:sum(event['assessment'].get('metrics',{}).get(metric,{}).get('reward',0) for event in events) for metric in METRICS}
    return {'redline':redline,'score':total['reward_score'],
            'status':'below_redline' if total['reward_score']<redline else 'healthy',
            'metrics':{metric:{'score':score,'status':'below_redline' if score<redline else 'healthy'} for metric,score in balances.items()},
            'recorded_iterations':len(events),
            'action':'Prioritize supported matched-control improvements; preserve failures and gates. Proxy/pending iterations earn zero. No score reset, fabricated reward, agent termination or deployment implied.'}


if __name__ == '__main__':
    base = dict.fromkeys(METRICS, .5)
    improved = dict.fromkeys(METRICS, .51)
    assert assess([improved], [base])['reward_delta'] == 4
    assert assess([base], [base])['reward_delta'] == -40
    assert assess([], [], eligible=False)['reward_delta'] == 0
    assert total_reward([])['reward_score'] == 0
    try:
        assess([dict.fromkeys(METRICS, float('nan'))], [base])
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid evidence accepted')
    print('Reward checks passed: improvement, ties, ineligible, initial and invalid inputs')
