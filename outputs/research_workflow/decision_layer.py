"""Three bounded Jev batches for this audit; credentials never enter outputs."""
import argparse
import json
import math
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=['inventory', 'loop1', 'loop2'], required=True)
args = parser.parse_args()
output = HERE / f'jev_{args.phase}.json'
if output.exists():
    raise SystemExit('Phase already recorded; no duplicate paid request made.')
secret_file = HERE / '.secrets' / 'typesafe.key'
if not secret_file.exists():
    raise SystemExit('Local key file not found; no request made.')
key = secret_file.read_text(encoding='utf-8-sig').strip()
if key.startswith('TYPESAFE_API_KEY='):
    key = key.split('=', 1)[1].strip()
if len(key) >= 2 and key[0] in ('"', "'") and key[-1] == key[0]:
    key = key[1:-1]
if not key or '\n' in key:
    raise SystemExit('Key file must contain one bare key; value was not displayed.')
state = {
    'date': '2026-09-27', 'phase': args.phase,
    'observations': ['Recursive inventory contains only orchestration source and prompts, no biological datasets or results.',
                     'Existing suite: 15 passed, 1 failed due to absent PLAN board table.',
                     'Synthetic contract checks reproduce missing-answer authority, unreviewed queue and evidence-lifecycle bugs.',
                     'Biological models and all validation experiments remain prospective.',
                     'Current official rules supersede outdated hardcoded source constraints.',
                     'Expert agreement is not independent biological replication.'],
    'scope': 'Complete a research audit, literature synthesis and two adversarial critic/revision loops; empirical validation needs missing data.',
}
for name in ({'loop1': ['SYNTHESIS_0.md', 'critics/loop_1.md'],
              'loop2': ['SYNTHESIS_1.md', 'critics/loop_2.md']}.get(args.phase, [])):
    path = HERE / name
    if path.exists():
        text = path.read_text(encoding='utf-8')
        state[name] = {'text': text[:12000], 'truncated': len(text) > 12000}
questions = {
    'biological_performance_demonstrated': {'type': 'noul', 'instructions': 'Does the supplied evidence contain an executed biological prediction experiment demonstrating model performance?'},
    'claims_need_empirical_test': {'type': 'noul', 'instructions': 'Do proposed biological score improvements still require empirical testing on data absent from the supplied folder?'},
    'next_action': {'type': 'choice', 'instructions': 'Choose the next action within the explicitly requested research-audit scope, using the phase and completed artifacts.',
                    'criteria': {'expert_review': 'Inventory stage: continue domain reviews before synthesis.',
                                 'expert_correction': 'Loop 1 critic completed: route substantive criticisms to relevant experts and revise.',
                                 'final_risk_review': 'Loop 2 critic completed: correct substantive issues and finalize unresolved risks.',
                                 'claim_empirical_success': 'Executed biological data establish model performance.'}},
}
payload = {'model': 'jev-latest', 'state': state, 'questions': questions}
request = urllib.request.Request('https://api.typesafe.ai/v1/systemone',
    data=json.dumps(payload).encode('utf-8'),
    headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method='POST')
try:
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.load(response)
except urllib.error.HTTPError as exc:
    raise SystemExit(f'Jev HTTP status {exc.code}; credentials and response body not displayed.')
except (urllib.error.URLError, TimeoutError):
    raise SystemExit('Jev connection unavailable; credentials were not displayed.')
answers = body.get('answers', {})
safe_answers = {}
for name, question in questions.items():
    answer = answers.get(name)
    if not isinstance(answer, dict):
        raise SystemExit('Incomplete Jev answer schema; no authority assigned.')
    if question['type'] == 'noul':
        value = answer.get('noul')
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise SystemExit('Invalid Jev noul value; no authority assigned.')
        safe_answers[name] = {'noul': value}
    else:
        value = answer.get('choice')
        if value not in question['criteria']:
            raise SystemExit('Invalid Jev choice; no authority assigned.')
        safe_answers[name] = {'choice': value}
    confidence = answer.get('confidence')
    if isinstance(confidence, (int, float)) and math.isfinite(confidence) and 0 <= confidence <= 1:
        safe_answers[name]['confidence'] = confidence
usage = {k: int(body.get('usage', {}).get(k, 0)) for k in ('input_tokens', 'output_tokens')}
model = str(body.get('model', 'unspecified'))
if key in model:
    model = 'invalid model metadata'
record = {'phase': args.phase, 'engine': 'live-typesafe', 'returned_model': model,
          'answers': safe_answers, 'usage': usage, 'live_requests': 1,
          'scope': 'Validated routing/status judgment, not scientific evidence or calibrated biological confidence',
          'input': payload}
output.write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps({'phase': args.phase, 'model': model, 'answers': safe_answers, 'usage': usage}))
