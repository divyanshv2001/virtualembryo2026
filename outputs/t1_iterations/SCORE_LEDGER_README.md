# Local score records

SCORE_LEDGER.jsonl indexes all observed local_score outcomes from preserved private report.json files and current partial reports, including invalid calibrations. Original reports and genuine events remain the source history. SCORE_LEDGER_MANIFEST.json records report hashes, indexed counts and any parse errors. Dataset preparation and diagnostic reports with no scored outcomes correctly have zero indexed rows.

Each record retains the observed headline, raw metric vector, calibrated skills, calibration floor/ceiling, candidate/fold/seed context, prediction hash when available and original report/plan/source provenance. Missing historical fields are null, not invented. Persistence raw metrics equal the measured same-report floor; this is the explicit baseline, not a simulated target. Invalid calibration records remain present. Official user-reported results stay in their separate source ledgers and outputs/t1_submissions/unit16_progress_20260928_01/official_result.json; local estimates are never official scores.

Refresh with outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/index_scores.py after every batch. Rebuilding this index does not overwrite original experiment histories. HORIZON_BATCH_RESULTS.json contains complete raw metrics, skills and generation audits for the newer one-day folds; its Markdown companion is a compact comparison.

Current coverage is unfinished. The authorized hourly continuation is recorded in RESEARCH_CONTINUATION.json. It preserves submission quota and continues declared viable methods and decisive ablations. Keep the computer on and desktop app running; scheduling is not evidence that a future run actually occurred.
