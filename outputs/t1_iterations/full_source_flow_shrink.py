"""Frozen source-latent flow shrinkage screen; never a challenge score."""
import json
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from train_extended_atlas import HERE
from cnf_density_flow import DensityFlowNet
from offline_backtest import load_core
from run_t1 import digest
from iterate import now, append_event


def main():
    pilot = HERE / 'private/full_source_flow_pilot_01'
    coords = HERE / 'private/full_source_latent_coordinates_01'
    out = HERE / 'private/full_source_flow_shrink_01'
    if out.exists():
        raise ValueError('Frozen screen already exists; preserve it')
    report = json.loads((pilot / 'report.json').read_text())
    if report['status'] != 'completed' or report['promotion_rule_passed']:
        raise ValueError('Requires completed failed pilot')
    out.mkdir()
    plan = {
        'created_utc': now(), 'source_sha256': digest(HERE / 'full_source_flow_shrink.py'),
        'pilot_report_sha256': digest(pilot / 'report.json'),
        'coordinates_sha256': digest(coords / 'coordinates.npz'),
        'checkpoint_sha256': report['checkpoints'],
        'alpha_grid': [0., .1, .25, .5, 1.], 'sources': ['sampled', 'full'],
        'cutoff': 8., 'targets': [8.25, 8.5], 'scorer_seed': 20260929,
        'metric': 'Pinned unbiased multi-kernel MMD in frozen 8D source latent space; lower is better',
        'rule': 'Advance only if one source/alpha strictly beats persistence at both held-out source stages; still requires full-panel historical and challenge-readiness checks.',
        'limits': 'Post-fit shrinkage uses no future expression for prediction. These two targets are development screens, not independent challenge validation or the official 32,285-gene score.',
        'submissions_allowed': 0, 'jev_requests_allowed': 0,
    }
    (out / 'plan.json').write_text(json.dumps(plan, indent=2))
    events = out / 'events.jsonl'
    append_event(events, 'plan_frozen', sha256=digest(out / 'plan.json'))
    with np.load(coords / 'coordinates.npz') as saved:
        z, stages, source_rows = (saved[k] for k in ('z', 'stages', 'source_rows'))
    donor_rows = np.load(pilot / 'donor_coordinate_rows.npy')
    donor_index = np.flatnonzero(np.isin(source_rows, donor_rows))
    if len(donor_index) != len(donor_rows) or not np.all(stages[donor_index] == 8.):
        raise ValueError('Frozen E8.0 donor mismatch')
    donor = z[donor_index].astype(np.float32)
    with np.load(HERE / 'private/cnf_feature_challenge_01/encoder4096.npz') as saved:
        basis, center = saved['basis'], saved['pca_center']
    models = {}
    for name in plan['sources']:
        checkpoint = pilot / (name + '.pt')
        if digest(checkpoint) != report['checkpoints'][name]:
            raise ValueError('Frozen checkpoint changed: ' + name)
        net = DensityFlowNet(basis, center, 8., 7.25)
        net.load_state_dict(torch.load(checkpoint, weights_only=False, map_location='cpu')['net'])
        net.eval()
        models[name] = net
    # Freeze every prediction before selecting the held-out target rows.
    predictions = {}
    with torch.no_grad():
        for target in plan['targets']:
            predictions[target] = {
                name: net.trajectory(torch.tensor(donor), [0., target - 8.], step=.125)[-1].numpy()
                for name, net in models.items()
            }
    core, _ = load_core()
    rows = []
    for target in plan['targets']:
        target_rows = np.load(pilot / f'target_coordinate_rows_{target}.npy')
        target_index = np.flatnonzero(np.isin(source_rows, target_rows))
        if len(target_index) != len(target_rows) or not np.all(stages[target_index] == target):
            raise ValueError('Frozen target mismatch')
        truth = z[target_index].astype(np.float32)
        control = float(core.mmd_unbiased(donor, truth, n=1000, n_pc=8, seed=plan['scorer_seed']))
        values = {}
        for name in plan['sources']:
            field_prediction = predictions[target][name]
            values[name] = {}
            for alpha in plan['alpha_grid']:
                candidate = donor + alpha * (field_prediction - donor)
                values[name][str(alpha)] = float(core.mmd_unbiased(candidate, truth, n=1000, n_pc=8, seed=plan['scorer_seed']))
        row = {'cutoff': 8., 'target': target, 'persistence_mmd': control, 'mmd_by_source_and_alpha': values}
        rows.append(row)
        append_event(events, 'source_shrinkage_scored', **row)
    passing = [
        {'source': name, 'alpha': alpha,
         'mmd': [r['mmd_by_source_and_alpha'][name][str(alpha)] for r in rows]}
        for name in plan['sources'] for alpha in plan['alpha_grid'] if alpha > 0 and
        all(r['mmd_by_source_and_alpha'][name][str(alpha)] < r['persistence_mmd'] for r in rows)
    ]
    result = {'status': 'completed', 'plan_sha256': digest(out / 'plan.json'),
              'pilot_report_sha256': plan['pilot_report_sha256'],
              'source_temporal_results': rows, 'passing_candidates': passing,
              'promotion_rule_passed': bool(passing), 'challenge_score': None,
              'official_score': None, 'full_panel_scoring': False,
              'conclusion': 'Source latent screen only; no official claim or full-panel promotion.'}
    (out / 'report.json').write_text(json.dumps(result, indent=2))
    public = {k: result[k] for k in ('source_temporal_results', 'passing_candidates',
                                    'promotion_rule_passed', 'challenge_score',
                                    'official_score', 'full_panel_scoring')}
    public.update(updated_utc=now(), plan_sha256=result['plan_sha256'],
                  report_sha256=digest(out / 'report.json'))
    (HERE / 'FULL_SOURCE_FLOW_SHRINK_RESULTS.json').write_text(json.dumps(public, indent=2))
    append_event(events, 'screen_completed', promotion_rule_passed=bool(passing))


if __name__ == '__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        main()
