# Agent Team evidence preparation

The screenshot requires trajectory plus prompts or harness. [Rules §9](https://virtualembryo.ai/challenge/rules) additionally require a fixed pre-run configuration and matching execution evidence; framework/model strings must be supplied. Evidence limits are 200 MB per file and 600 MB total per team. Prize verification can require all three kinds.

`package_evidence.py` exports recorded visible events from this project's original Codex root/delegated sessions, preserving their order and timestamps. It does not invent a retrospective run. Every source has a checksum and disclosure of exported/omitted record counts; credentials are redacted. Private framework reasoning and privileged instructions are excluded, as are unrelated account/guardian sessions. The official [OpenAI Docs event-log example](https://developers.openai.com/blog/eval-skills) explains prospective `codex exec --json` capture; the current app session export uses the actual local log format instead.

Generated local files under `private/`:

- `trajectory.zip`: recorded visible actions, results, messages and their provenance.
- `prompts.zip`: recorded visible user/delegation prompt events plus the adopted agenticprompt document; excludes platform-controlled instructions.
- `harness.zip`: actual T1 task tools, public contracts, dependency pins and contemporaneous supporting run report; not the full Codex framework or a custom episode runner.

These archives contain project conversations and remain ignored in Git. No upload occurs. Export completion/ZIP integrity does not certify acceptance, prompt completeness or autonomous-track eligibility. The current interactive development session has **no recorded pre-run configuration lock**. Do not claim that these files prove a locked Agent Team run. Use organizer guidance or produce a new properly configured run with prospective capture; do not backdate a lock, substitute another run's evidence or manually alter predictions. A future setup must fix prompts/model/tools/permissions/budget before execution and allow the agent to select its final result without result-driven human steering.

The locally recorded model string is obtained from turn_context events, not guessed from assistant branding. See private/export_provenance.json for actual framework/model/source records and archive hashes. Keep the original framework logs locally; sanitization disclosures accompany the exports.
