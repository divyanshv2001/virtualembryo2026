# Token-efficient Jev policy

The three approved audit requests used 9,094 actual input tokens and 316 output tokens. This historical usage remains unchanged. The previous authorization is exhausted; this preparation performs **zero additional live calls** and does not transmit data or source files.

1. Deterministic code handles matrix schema, quotas, counts, file hashes, missing evidence and explicit phase transitions. The current three audit routing decisions are directly inferable, so future reruns should skip Jev entirely.
2. Use Jev for genuinely ambiguous bounded choices: routing a new cross-disciplinary issue, relevance classification or deduplication requiring semantics. It cannot establish biological truth, file validity or resource eligibility.
3. Supply concise structured facts, uncertainty and issue IDs; retain full evidence in local artifacts for reasoning/review. Do not repeatedly send both synthesis and full critic prose for simple routing. Never silently truncate a substantive objection to obtain a smaller prompt.
4. Batch independent questions sharing one state. Dependent questions require later steps. The [official batching cookbook](https://docs.typesafe.ai/cookbooks/parallel_questions) demonstrates lower cost on its specific workload; its measured speed/cost ratios are not guarantees for this project.
5. Pin the tested model version for routing semantics, hash canonical state plus complete question meanings and schema/model versions. Only validated complete live answers may enter a decision cache. Missing answers, malformed probabilities and local heuristics are never authoritative. Changed evidence/questions invalidate reuse.
6. Use a preflight byte cap and a separately approved request/attempt budget; count failed/retried attempts too. Actual API usage supplies token accounting. Byte counts alone cannot prove tokenizer savings. Low confidence or scientific disagreement escalates to relevant experts with just linked evidence, not all 16 personas.

`compact_jev.py` prepares offline counterfactual routing packets and semantic cache keys. It contains no HTTP client and no credential access. `token_plan.json` measures serialized-byte reduction against historical requests; optimized API token usage remains unknown. It is an offline planning/validation helper, not a live caching implementation or scientific critic replacement. Typed contract checks match the documented [API primitives](https://docs.typesafe.ai/api).

Run from workspace:

```powershell
python outputs/submission_preparation/compact_jev.py
python -m unittest discover -s outputs/submission_preparation -p test_compact_jev.py
```

Before extending live Jev use, specify new allowed payload categories, destination, maximum attempts and budget. Keep keys in the existing ignored secret file. A new payload hash is not new permission to send protected data.
