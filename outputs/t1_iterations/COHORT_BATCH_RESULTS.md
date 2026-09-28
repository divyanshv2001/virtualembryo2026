# Completed source-cohort batch

All three seeds and all four floor/ceiling calibration anchors exactly match the previous state-search panels. The 27-evaluation broad-cohort grid has best mean 51.596 (unit_k8). The 24-evaluation source-importance batch has best new mean 51.629 (25% adaptation, 8 states); the replayed incumbent reproduces mean 55.312 exactly. No new variant is promoted.

Importance sampling uses observed E8.5 donor states and atlas rows <=E8.5 only. It selects unique rows without replacement and applies the same bounded source-to-anchor state ratio across past stages. This avoids duplicate-cell inflation but can distort past demographic trends; the resulting sampled composition does not identify biological growth.

COHORT_BATCH_RESULTS.json retains every candidate, all raw metric results, mean four-metric skill vectors and report hashes. The local >72 gate remains unmet. No official submission or Jev request was made by these experiments. The separate user-requested E10.5 progress export is an export override, not research promotion.
