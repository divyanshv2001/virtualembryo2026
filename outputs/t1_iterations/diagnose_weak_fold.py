"""Describe errors on an already-used development fold; never train on its target."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from constrained_forecast import PopulationForecast
from train_extended_atlas import HERE
from iterate import append_event, now
from run_t1 import digest


def main():
    out = HERE/'private/weak_fold_audit_01'
    if out.exists(): raise ValueError('Preserve previous audit')
    data = HERE/'private/cardiac_prepared_01'
    previous = HERE/'private/cardiac_global_01/report.json'
    prior = json.loads(previous.read_text())
    prepared = json.loads((data/'report.json').read_text())
    for name, key in [('expression.npy', 'expression_sha256'), ('selected_metadata.csv', 'metadata_sha256')]:
        if digest(data/name) != prepared[key]: raise ValueError('Input changed')
    out.mkdir()
    events = out/'events.jsonl'
    (out/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    plan = {'created_utc':now(), 'cutoff':8.25, 'target':8.5,
        'candidate':'tilt_0.5_expr_0.1', 'source_report_sha256':digest(previous),
        'scope':'Descriptive development error analysis; not new validation or proof of lineage.',
        'code_sha256':digest(Path(__file__))}
    (out/'plan.json').write_text(json.dumps(plan, indent=2))
    append_event(events, 'diagnostic_plan_frozen', sha256=digest(out/'plan.json'))
    x = np.load(data/'expression.npy', mmap_mode='r')
    metadata = pd.read_csv(data/'selected_metadata.csv')
    stages = metadata.numeric_stage.to_numpy(float)
    model = PopulationForecast(x, stages, 8.25)
    prediction, _, indices, _ = model.predict(8.5, .5, .1)
    # Target expression is first accessed for diagnostics after prediction generation.
    reference = np.asarray(x[stages == 8.25]); truth = np.asarray(x[stages == 8.5])
    labels = metadata.celltype_extended_atlas.to_numpy(str)
    ref_labels = labels[stages == 8.25]; pred_labels = ref_labels[indices]
    truth_labels = labels[stages == 8.5]
    groups = []
    for label in sorted(set(ref_labels)|set(truth_labels)):
        a, b, t = ref_labels == label, pred_labels == label, truth_labels == label
        row = {'label':label, 'reference_cells':int(a.sum()), 'predicted_donors':int(b.sum()), 'target_cells':int(t.sum()),
            'floor_proportion_error':float(a.mean()-t.mean()), 'prediction_proportion_error':float(b.mean()-t.mean())}
        if min(a.sum(), b.sum(), t.sum()) >= 15:
            mean = truth[t].mean(0, dtype=float)
            row.update(floor_mean_rmse=float(np.sqrt(np.mean((reference[a].mean(0, dtype=float)-mean)**2))),
                prediction_mean_rmse=float(np.sqrt(np.mean((prediction[b].mean(0, dtype=float)-mean)**2))))
        groups.append(row)
    fold = next(f for f in prior['folds'] if f['training_cutoff'] == 8.25)
    result = next(r for r in fold['results'] if r['candidate'] == plan['candidate'])
    contributions = {k:100*w*(result['skills'][k]-.5)
        for k, w in [('de_score', .25), ('de_direction', .25), ('mmd_u', .3), ('variogram', .2)]}
    states = np.load(HERE/'private/cardiac_state_01/states_8.0.npz')
    report = {'plan':plan, 'metric_contributions_vs_50':contributions, 'groups':groups,
        'state_support_at_cutoff_8':states['state_support'].tolist(),
        'unsupported_states_at_cutoff_8':int((states['state_support'] < 8).sum()),
        'population_total_variation_floor':float(sum(abs(g['floor_proportion_error']) for g in groups)/2),
        'population_total_variation_prediction':float(sum(abs(g['prediction_proportion_error']) for g in groups)/2)}
    (out/'report.json').write_text(json.dumps(report, indent=2))
    append_event(events, 'weak_fold_diagnosis_completed', report_sha256=digest(out/'report.json'))


if __name__ == '__main__':
    with threadpool_limits(limits=2): main()
