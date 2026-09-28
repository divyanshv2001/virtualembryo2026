"""Explicit status amendment preserving evidence identifiers and expert claims."""
import json
from pathlib import Path
here = Path(__file__).resolve().parent
if (here/'SYNTHESIS_2.md').exists():
    raise SystemExit('Final loop amendments exist; historical loop-1 amendment must not overwrite them.')
path = here/'hypothesis_registry.json'
rows = json.loads(path.read_text())
for row in rows:
    row['status'] = 'untested'
    row['status_basis'] = 'No measured biological prediction experiment supplied or executed.'
    row['claim_scope_review'] = 'PROCESS board labels are insufficient for automatic biological scope classification; retain original proposal and inspect claim individually.'
path.write_text(json.dumps(rows,indent=2),encoding='utf-8')
path = here/'critics/issues_1.json'
rows = json.loads(path.read_text())
for row in rows:
    row['status'] = 'design_corrected_empirical_or_operational_risk_open'
    row['resolution_artifact'] = 'critics/resolution_1.md'
path.write_text(json.dumps(rows,indent=2),encoding='utf-8')
print('28 proposal statuses remain untested; critic dispositions recorded separately.')
