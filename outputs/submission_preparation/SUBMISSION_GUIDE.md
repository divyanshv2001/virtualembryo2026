# Submission preparation — verified 2026-09-28

The research report is not a prediction upload. Prepare a board-specific AnnData `.h5ad` artifact, then use the signed-in team submission form. The public view says a team must be registered; it cannot establish the user's login/team status or expose the authenticated form. No upload/API endpoint is inferred or used. [Official submission page](https://virtualembryo.ai/challenge/account/submissions).

## Upload contract

| Field | Required behavior |
|---|---|
| `var_names` | Exact board genes in released order; use the official board panel, not an arbitrary training file |
| `.X` | Two-dimensional cells × genes, finite and nonnegative log-normalized expression; sparse or dense accepted, converted to float32 |
| `obsm['spatial_3D']` | Required for T2/T3: cells × at least 3 finite columns; first three are used. T1 carries no coordinates |
| `obs['celltype']` | Optional and ignored; scorer supplies its own types |

Validation failure rejects the file before scoring. `--allow-reorder` is mentioned for the scorer; do not assume an equivalent web-form control exists. Export in exact order. [Official upload checks](https://virtualembryo.ai/challenge/account/submissions).

The historical local validator rejects every `obs` column and requires exactly three coordinate columns; those restrictions are stricter than this public contract. Its unavailable dependencies also prevent execution here. It is not the verified web uploader.

## Board selection

| Development board | Genes | Allowed cells |
|---|---:|---:|
| T1:val | 32,285 | 1,000–5,118 |
| T2:embryo:val_interp | 498 | 583–5,000 |
| T2:heart:val_interp | 500 | 1,000–17,616 |
| T2:heart:val_extrap | 500 | 1,000–25,179 |
| T3:gata4 | 500 | 1,000–7,449 |

Obtain the panel and machine-readable index linked from [official data](https://virtualembryo.ai/challenge/data). Recheck current board/phase before exporting. Cell count is sampling support, not predicted embryo size. Coordinates are per-embryo local and unregistered across time; row pairing cannot justify pooled anatomical interpretation. These are public format facts, not evidence that this project has a valid prediction.

## Concrete preparation sequence

1. Sign in/register the intended team; select task, setting, target and current phase.
2. Obtain eligible training files, official panel/index and a reproducible model/provenance manifest.
3. Fit only eligible inputs; produce predicted expression and meaningful required coordinates under a separately verified spatial contract.
4. Export `.h5ad`; validate panel/order, shape, bounds, finite/nonnegative expression and required coordinate shape/values. Record artifact checksum and provenance.
5. Upload through the authenticated page, retain validation/scoring records, and obey current submission quotas and resource disclosure requirements.

Current blocker: no local biological data, trained predictor or prediction file. A prospective E7.25 local proxy is not the official T2 E7.5 validation prediction. Competition eligibility also remains unresolved after the audit's literature exposure; do not treat format validation as eligibility approval.

## Community resources

The [resources page](https://virtualembryo.ai/challenge/resources) describes reviewed and unreviewed participant contributions. At inspection no actual contribution links were rendered under either category. A listed unreviewed resource would not imply organizer endorsement. If a particular resource is intended, record its exact URL and review its code/provenance before use. The prediction upload and community-contribution award are separate flows.
