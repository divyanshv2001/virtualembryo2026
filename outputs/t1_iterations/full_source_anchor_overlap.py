"""Past-only full-atlas support audit for E8.5 challenge anchors."""
from collections import Counter
import json
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event


def main():
    root = HERE.parents[1]
    old = HERE / 'private/cnf_feature_challenge_01'
    coordinates = HERE / 'private/full_source_latent_coordinates_01'
    out = HERE / 'private/full_source_anchor_overlap_01'
    if out.exists():
        raise ValueError('Preserve frozen overlap audit')
    anchor = root / 'data/E8.5_RNA.h5ad'
    panel = (root / 'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
    with np.load(old / 'encoder4096.npz') as encoder:
        features = encoder['features']; center = encoder['center']; scale = encoder['scale']
        pca_center = encoder['pca_center']; basis = encoder['basis']
    with np.load(coordinates / 'coordinates.npz') as saved:
        z = saved['z']; stages = saved['stages']; source_rows = saved['source_rows']
    meta = pd.read_csv(HERE / 'private/extended_atlas/meta.tsv', sep='\t',
                       usecols=['celltype_extended_atlas'], low_memory=False)
    source_mask = stages == 8.5
    source = z[source_mask]
    source_labels = meta.celltype_extended_atlas.fillna('unannotated').to_numpy(str)[source_rows[source_mask]]
    plan = {
        'created_utc': now(), 'code_sha256': digest(HERE / 'full_source_anchor_overlap.py'),
        'coordinates_sha256': digest(coordinates / 'coordinates.npz'),
        'encoder_sha256': digest(old / 'encoder4096.npz'),
        'anchor_sha256': digest(anchor), 'panel_sha256': digest(root / 'outputs/t1_run/T1__val.genes.txt'),
        'source_stage': 8.5, 'challenge_stage': 8.5, 'neighbors': 16,
        'support_rule': 'A challenge cell has strict support only if its first full-source neighbor is <=2x the median full-source leave-one-out nearest-neighbor distance, at least 8/16 neighbors share their modal source label, and that label exactly matches the challenge celltype. Exact string identity is a conservative diagnostic, not a validated ontology.',
        'decision_rule': 'Do not use direct nearest-neighbor source transfer if strict support is below 10% of all anchors or no challenge type with >=100 cells has >=50% strict support.',
        'scope': 'Read-only, past E8.5 only; no E9.5 expression, forecasts, training, official score or submission.',
    }
    out.mkdir(); (out / 'plan.json').write_text(json.dumps(plan, indent=2))
    events = out / 'events.jsonl'; append_event(events, 'plan_frozen', sha256=digest(out / 'plan.json'))
    a = ad.read_h5ad(anchor, backed='r')
    try:
        if not a.var_names.is_unique or a.var_names.tolist() != panel:
            raise ValueError('Challenge panel mismatch')
        challenge = np.empty((a.n_obs, len(basis)), dtype=np.float32)
        challenge_labels = a.obs.celltype.fillna('unannotated').to_numpy(str)
        for start in range(0, a.n_obs, 256):
            end = min(start + 256, a.n_obs)
            block = a.X[start:end, features]
            values = block.toarray() if sparse.issparse(block) else np.asarray(block)
            challenge[start:end] = ((values.astype(np.float32)-center)/scale-pca_center) @ basis.T
    finally:
        a.file.close()
    with threadpool_limits(limits=2):
        nn = NearestNeighbors(n_neighbors=16, algorithm='kd_tree').fit(source)
        d_self, _ = nn.kneighbors(source, n_neighbors=2)
        d_anchor, ids = nn.kneighbors(challenge)
    reference = float(np.median(d_self[:, 1]))
    nearest_labels = source_labels[ids]
    modal = []
    agreement = np.empty(len(challenge), dtype=np.float32)
    for i, labels in enumerate(nearest_labels):
        label, count = Counter(labels).most_common(1)[0]
        modal.append(label); agreement[i] = count / 16
    modal = np.asarray(modal)
    ratio = d_anchor[:, 0] / max(reference, 1e-8)
    geometric = (ratio <= 2.) & (agreement >= .5)
    strict = geometric & (modal == challenge_labels)
    by_type = []
    for label in sorted(np.unique(challenge_labels)):
        mask = challenge_labels == label
        by_type.append({'challenge_type': label, 'cells': int(mask.sum()),
                        'median_distance_ratio': float(np.median(ratio[mask])),
                        'geometric_support_fraction': float(geometric[mask].mean()),
                        'strict_exact_label_support_fraction': float(strict[mask].mean())})
    sufficient = [x for x in by_type if x['cells'] >= 100 and x['strict_exact_label_support_fraction'] >= .5]
    decision = not (float(strict.mean()) < .1 or not sufficient)
    report = {'status': 'completed', 'plan_sha256': digest(out / 'plan.json'),
              'source_rows': len(source), 'challenge_rows': len(challenge),
              'source_leave_one_out_median': reference,
              'challenge_median_distance_ratio': float(np.median(ratio)),
              'distance_ratio_le_two_fraction': float((ratio <= 2.).mean()),
              'geometric_support_fraction': float(geometric.mean()),
              'strict_exact_label_support_fraction': float(strict.mean()),
              'exact_label_overlap': sorted(set(source_labels) & set(challenge_labels)),
              'types_with_majority_strict_support': sufficient,
              'by_challenge_type': by_type,
              'direct_nearest_source_transfer_allowed': bool(decision),
              'limits': 'Exact celltype names are not an ontology alignment. This is a support screen, not model accuracy or a challenge score.'}
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    public = dict(report, created_utc=now(), report_sha256=digest(out / 'report.json'))
    (HERE / 'FULL_SOURCE_ANCHOR_OVERLAP.json').write_text(json.dumps(public, indent=2))
    append_event(events, 'overlap_audited', strict_support=report['strict_exact_label_support_fraction'],
                 direct_transfer_allowed=bool(decision))


if __name__ == '__main__':
    main()
