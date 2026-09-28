"""Persistent cross-examination registry, based on saved two-sided discussions."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = [
    ('D1','devbio','neuralode','Snapshot forecast versus cellular fate',
     'mmd_u/de_direction can assess forecast marginals; ancestry needs tracing.',
     'Accepts marginal-only utility; numerical convergence does not identify dynamics.',
     'Lineage correspondence, independent growth/death and replicated embryos.',
     'Compare equally accurate marginal models against withheld clonal descendants and independent abundance.',
     'resolved by reasoning on claim scope; biological dynamics requires experiment'),
    ('D2','scgenomics','spatialtx','Order bias versus subset neighborhood support',
     'Seeded draws do not fix rebuilding 15-neighbor means on sparse subsets.',
     'Accepts separate full-cloud-profile versus rebuilt-neighborhood comparison.',
     'Scorer order of subsampling, imaging/segmentation and physical units.',
     'Cross row order, subset size and full versus rebuilt neighborhoods; isolate within-type pairing.',
     'unresolved but narrowed; requires scorer/data experiment'),
    ('D3','grn','causal','Predictive priors versus identified causal regulation',
     'Eligible graph identities can predict without identified direct edges.',
     'Accepts predictive utility; randomization controls do not resolve causal direction or unseen interventions.',
     'Independent permitted interventions, genotype/background overlap and prior provenance.',
     'Matched true/randomized priors on unseen permitted interventions, plus independent causal intervention study.',
     'resolved by reasoning on prediction/identification; transfer requires experiment'),
    ('D4','sysbio','sciml','Capture versus physical abundance',
     'Documented observation operator cannot identify capture and abundance from rows alone.',
     'Accepts abstention for equivalent worlds; calibrated auxiliary counts may resolve ambiguity.',
     'Capture calibration and whole-tissue/state counts.',
     'Equivalent two-world diagnostic then independent calibrated in situ counts/recovery controls.',
     'resolved by counterexample on nonidentifiability; biology requires measurement'),
    ('D5','genmodel','diffusion','Flow benefit beyond a strong conditional residual sampler',
     'Mean-only trends are weak controls; preserve conditional covariance in simple comparators.',
     'Accepts covariance comparator and matched mixture weights; endpoint gain cannot validate paths.',
     'Eligible stage/embryo series, scorer, support and independent abundance.',
     'Matched-budget flow versus conditional residual sampler, coupling sensitivity and separate horizons.',
     'unresolved but narrowed; requires predictive experiment'),
    ('D6','gnn','spatial3d','Graph/scorer topology versus anatomical structure',
     'kNN and local average descriptors can miss lumen crossings or partial coverage.',
     'Accepts separate segmentation/coverage/orientation endpoints and synthetic shells.',
     'Independent 3D structural masks, acquisition coverage, units and laterality.',
     'Known-shell graph tests plus calibrated imaging landmarks on eligible specimens.',
     'unresolved but narrowed; anatomy requires measurements'),
    ('D7','replearn','biovalid','Representation benefit versus circular panel or decoder validation',
     'Marker scoring on the same panel is correlated validation; decoder effects need matching.',
     'Accepts shared/crossed decoder controls and independent protein/domain observations.',
     'Eligible replicate manifest, assay availability and prespecified unknown-state thresholds.',
     'Cross encoders/decoders; held-out conditional states and blinded protein imaging/counts.',
     'unresolved but narrowed; orthogonal benefit requires experiment'),
    ('D8','stats','benchdesign','Frozen primary claim, guardrails and independent assessment',
     'Superiority needs δ*; no-harm needs explicit η_j and simultaneous claims; selection consumes target.',
     'Accepts provisional T2 embryo interpolation proxy and composition-only primary comparator.',
     'Numerical margins, replicate/litter design and untouched assessment resource.',
     'Freeze E6.75/E8.0→E7.25 proxy, pilot margins and query ledger before independent assessment.',
     'resolved on protocol logic; numerical/replicate gates remain unresolved'),
]
registry = []
for id_, a, b, issue, ap, bp, missing, experiment, conclusion in rows:
    if not all((HERE/'debate'/f'{x}.md').exists() for x in (a,b)):
        raise SystemExit('Incomplete two-sided discussion: '+id_)
    registry.append({'id': id_, 'issue': issue, 'expert_a': a, 'expert_a_position': ap,
                     'expert_b': b, 'expert_b_position': bp,
                     'evidence_a': f'debate/{a}.md; experts/{a}.md',
                     'evidence_b': f'debate/{b}.md; experts/{b}.md',
                     'missing_evidence': missing, 'resolving_experiment': experiment,
                     'current_provisional_conclusion': conclusion,
                     'note': 'Attributed actual artifact exchange; agreement is not empirical replication.'})
(HERE/'disagreement_registry.json').write_text(json.dumps(registry,indent=2),encoding='utf-8')
print('Eight two-sided issue records saved.')
