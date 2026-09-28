"""Reconstruct the rejected saved rollout and measure its distribution changes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from train_extended_atlas import DATA, HERE, normalize_log
from constrained_forecast import distribution_audit
from iterate import append_event
from run_t1 import digest


def main():
    old = HERE/'private/extended_training_01'
    out = HERE/'private/rollout_audit_01'
    if out.exists(): raise ValueError('Preserve audit history')
    prepared = json.loads((DATA/'report.json').read_text())
    if digest(DATA/'expression.npy') != prepared['expression_sha256']: raise ValueError('Input changed')
    out.mkdir()
    events = out/'events.jsonl'
    (out/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    checkpoint = old/'final_checkpoint.npz'
    append_event(events, 'audit_started', checkpoint_sha256=digest(checkpoint), code_sha256=digest(Path(__file__)))
    m = np.load(checkpoint)
    config = json.loads((old/'report.json').read_text())['selected']
    stages = pd.read_csv(DATA/'selected_metadata.csv').numeric_stage.to_numpy(float)
    x = np.load(DATA/'expression.npy', mmap_mode='r')
    donor = np.asarray(x[stages == 8.5])
    scaled = np.clip((donor[:, m['features']]-m['center'])/m['scale'], -10, 10)
    z = (scaled-m['pca_mean']) @ m['pca_components'].T / np.sqrt(m['pca_explained_variance'])
    original = z.copy()
    for t in [8.5, 8.75, 9., 9.25]:
        time = np.full((len(z), 1), t-7.)
        phi = np.concatenate([z, time, z*time], axis=1)
        v = phi @ m['model_0_coef'].T + m['model_0_intercept']
        v *= np.minimum(1, m['speed_cap']/np.maximum(np.linalg.norm(v, axis=1), 1e-9))[:, None]
        z += .25*config['strength']*v
    preclip = donor.astype(float) + (z-original) @ m['decoder']
    prediction = normalize_log(preclip)
    features = m['features'][:128]
    report = {'source_checkpoint_sha256':digest(checkpoint), 'config':config,
        'donor':distribution_audit(donor, donor, features),
        'rejected_rollout':distribution_audit(prediction, donor, features),
        'target_diagnostic_only':distribution_audit(np.asarray(x[stages == 9.5]), donor, features),
        'preclip_negative_fraction':float((preclip < 0).mean()),
        'interpretation':'Descriptive diagnostics of an already evaluated forecast. Target diagnostics never enter forecast generation. Causes are not uniquely identified.',
        'submissions_used':0, 'jev_requests_used':0}
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    append_event(events, 'audit_completed', report_sha256=digest(out/'report.json'))


if __name__ == '__main__':
    with threadpool_limits(limits=2): main()
