"""Check or sync the two authorized prompt updates into the ignored source checkout."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(content): return hashlib.sha256(content.replace(b'\r\n', b'\n')).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent; root = here.parents[1]
    record = json.loads((here/'PROMPT_UPDATE.json').read_text())
    entries = [
        ('Neurl IPS 2026/agenticprompt', here/'AGENTICPROMPT_UPDATED.md', record['source_before_normalized_sha256'], record['updated_sha256']),
        ('Neurl IPS 2026/orchestration/diverge.py', here/'diverge.updated.py.txt', record['generator_before_sha256'], record['generator_sha256'])]
    checked = []
    # Validate both targets before writing either one. Unknown local edits are preserved.
    for relative, canonical, before, after in entries:
        target = root/relative; payload = canonical.read_bytes()
        if sha(payload) != after: raise ValueError('Canonical prompt checksum mismatch')
        current = sha(target.read_bytes())
        previous = record.get('previous_updated_sha256' if relative.endswith('agenticprompt') else 'previous_generator_sha256', [])
        if current not in [before, after]+previous: raise ValueError('Unrecognized local changes: '+relative)
        checked.append((target, payload, current != after))
    if args.apply:
        for target, payload, changed in checked:
            if changed: target.write_bytes(payload)
    print(json.dumps({'apply':args.apply, 'files':[{'path':p.relative_to(root).as_posix(),
        'needed_update':changed, 'matches_updated':sha(p.read_bytes())==sha(payload)}
        for p, payload, changed in checked]}, indent=2))


if __name__ == '__main__': main()
