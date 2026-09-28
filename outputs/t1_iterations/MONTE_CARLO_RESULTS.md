# Prompt update and Monte Carlo stability pilot

The user-requested persistent >72 objective is now explicit in [LOCAL_OPTIMIZATION_PROMPT.md](LOCAL_OPTIMIZATION_PROMPT.md) and [LOCAL_OPTIMIZATION_STATE.json](LOCAL_OPTIMIZATION_STATE.json). A completed batch does not complete the objective. Future work must revise weak models, retain failures, test all four metrics and avoid favorable-seed selection or changed calibration.

The original `Neurl IPS 2026/agenticprompt` has an appended continuation mandate, preserving its historical content. Its Git-tracked copy is [AGENTICPROMPT_UPDATED.md](../research_workflow/AGENTICPROMPT_UPDATED.md). The original prompt generator now adds a compact pointer to the mandate and checkpoint. Sixteen updated expert prompt files were generated under `prompts/`, with maximum size 7,605 bytes. They are preparation artifacts: **no agents were invoked and no additional Jev request was made**. The separately versioned source folder remains ignored by the root repository; [PROMPT_UPDATE.json](../research_workflow/PROMPT_UPDATE.json) records source/canonical hashes and a tracked generator snapshot.

After pulling this root repository, `outputs/research_workflow/.venv/Scripts/python.exe outputs/research_workflow/sync_optimization_prompts.py --apply` can sync the two tracked snapshots into the local source checkout. It validates both inputs before writing and rejects unrecognized local edits. Check mode verified that both updated local files already match their canonical snapshots.

## Executed Monte Carlo pilot

`monte_carlo_stability.py` evaluates the frozen direct-projection eight-state candidate and incumbent against matched persistence. All use the complete 32,285-gene panel. Sixteen seeds were frozen before scoring. Each replicate bootstraps 1,500 paired reference/prediction row positions and draws 2,000 observed challenge E9.5 cells, split into 1,000 truth and 1,000 ceiling cells. Each candidate shares that replicate's reference, truth, ceiling and scorer calibration.

Predictions are fixed. This simulation measures conditional sampling stability; it does not train a better predictor, create independent embryos, or validate hidden E10.5. The candidate was selected on earlier development panels, and the biological stage remains reused development data. Do not interpret the empirical interval as an independent biological confidence interval.

| Frozen candidate | Mean local score | Empirical 2.5th–97.5th percentiles | Mean gain over persistence |
|---|---:|---:|---:|
| Eight-state, expression 0.25, direct projection | **50.92** | 50.55–51.27 | +0.92 |
| Incumbent global | 49.89 | 47.72–51.72 | −0.11 |

All 16 calibrations passed. The best candidate's empirical lower-tail paired gain is +0.55, but the absolute score is far below 72. The incumbent's lower-tail paired gain is −2.28. Neither passes the final gate, and a 16-replicate pilot cannot satisfy the mandated >=64-replicate final check even if its mean were favorable. Further Monte Carlo draws of this fixed weak predictor should not be used as a score-improvement strategy.

The 16 replicates produced 32 candidate evaluations plus 16 matched persistence controls. Two decision tests passed: a lucky maximum/short pilot cannot pass promotion, and invalid calibration cannot be discarded to create a passing score. Actual partial reports, sampling indices, plan/source hashes and event logs are retained under ignored `private/monte_carlo_pilot_01`. Each completed replicate writes a resume checkpoint. A completed-run `--resume` check reproduced the report without repeating trials.

```powershell
outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/monte_carlo_stability.py --round NEW_PILOT --replicates 16
outputs/research_workflow/.venv/Scripts/python.exe outputs/t1_iterations/monte_carlo_stability.py --round NEW_PILOT --replicates 16 --resume
outputs/research_workflow/.venv/Scripts/python.exe -m unittest discover -s outputs/t1_iterations -p test_monte_carlo_stability.py
```

Changing a frozen seed list, replicate count, source code or input hash requires a new run. Resume skips completed replicates rather than overwriting history. The next scientific step recorded in the state file is a new support/detection-aware reconstruction hypothesis; it is **not yet implemented or demonstrated**. The optimization objective remains `threshold_not_met`; no process is currently running and no official submission has been used.

## Usage-limit recovery

The mandate requires checkpointing, reading an actually exposed reset timestamp, waiting/retrying when the active runtime supports it, and recording unknown reset times honestly. Codex documentation identifies the usage dashboard for current limits/reset times and `/status` for remaining limits in a CLI session. No account-specific quota/status tool is available in this session, and no limit event or reset timestamp was observed. The saved reset value is therefore unknown. No automatic wake-up service was installed or claimed. [Official Codex documentation](https://learn.chatgpt.com/docs/pricing).

Temporary API rate limits use observed retry/reset information and bounded backoff; billing/credit exhaustion is a different condition and must not trigger indefinite retries. No paid API switch, credit purchase, account change or new external routing is authorized by these prompt changes. [Official API retry guidance](https://developers.openai.com/api/docs/guides/rate-limits).
