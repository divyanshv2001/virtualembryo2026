# Executable T1 validation baselines

Targets official `T1:val` (E10.5), using local released E8.5/E9.5 RNA inputs only. No held-out target is read, no API call is made and no upload occurs. Original source and data are preserved.

## Outputs

- `T1_val__sampled_copy_last.h5ad`: 2,000 seeded E9.5 rows sampled without replacement, unchanged expression in official gene order. Fresh prediction IDs, no copied annotations/UMAP/coordinates/metadata. This empirical sample is not exactly the official baked floor and has no measured official score.
- `T1_val__pseudobulk_shift_exploratory.h5ad`: optional public-style comparator. For exact matched training cell-type labels, adds E9.5 mean minus E8.5 mean to those same E9.5 donor rows and clips negatives at zero. Unmatched last-stage types stay unchanged. This uses cell-weighted training means, not an independent-embryo model; annotation harmonization and future-stage improvement are unverified. Clipping changes expression moments; no extra normalization is inferred.
- `run_report.json`: local checksums, versions, parameters and output validations. `input_inspection.json` and `donor_rows.npy` retain local inspection/donor records. These data-derived records and all biological `.h5ad` files stay ignored in Git.

`copy_last` and `pseudobulk_shift` are described in the [official reference rows](https://virtualembryo.ai/challenge/baselines). Implementations here use a fixed sampled donor set to fit submission bounds and make a matched comparison; no exact starter-kit/scorer parity is claimed. Current two snapshots cannot validate two-stage extrapolation by holding one stage out. An exploratory file is not evidence of improvement; use persistence as the conservative initial upload candidate.

## Execute

From workspace, with an environment containing `anndata`, `h5py`, `numpy`, `pandas` and `scipy`:

```powershell
python outputs/t1_run/fetch_contract.py
outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_run/inspect_inputs.py
outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_run/run_t1.py --cells 2000 --seed 20260928 --shift-candidate
outputs/research_workflow/.venv/Scripts/python.exe -m unittest discover -s outputs/t1_run -p test_t1.py
```

Public panel/index provenance and checksums are in contract_sources.json; canonical newline-joined panel hash is checked against the index. Dependencies used for this execution are pinned in requirements-lock.txt. Fetching contracts requires network; model/validation execution is local. Re-execution replaces generated local output files, not input data.

Validation covers official panel/order, current cell bounds, float32, finite nonnegative `.X`, absence of T1 coordinates and persistence row identity after round-trip. It does not certify future expression accuracy, full assay normalization semantics, official scores or competition eligibility. Upload only through your registered team's [submission page](https://virtualembryo.ai/challenge/account/submissions), with required provenance/disclosures and track rules. This script does not supply a competition track eligibility verdict.

Jev usage for this run is zero: format/hash checks and the prescribed baseline formulas are deterministic. Historical research registries remain snapshots of the earlier no-data audit; this new run's local report records the data now supplied and executed baselines.
