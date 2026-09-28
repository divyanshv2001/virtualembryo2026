"""Fetch pinned public metric source only; never fetch bundled biological data."""
import hashlib
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = HERE / 'private' / 'scorer_source'
FILES = ['LICENSE', 'README.md', 'common/core_metrics.py', 'T1/metrics.py', 'pyproject.toml']


def get(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'VirtualEmbryoResearch/1.0'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = DEST / 'manifest.json'
    if manifest.exists():
        m = json.loads(manifest.read_text())
        for r in m['files']:
            if hashlib.sha256((DEST/r['path']).read_bytes()).hexdigest() != r['sha256']:
                raise ValueError('Pinned source changed')
        print(json.dumps({'commit': m['commit'], 'files': len(m['files']), 'cached': True}))
        return
    commit = json.loads(get('https://api.github.com/repos/aristoteleo/veckit/commits/main'))['sha']
    records = []
    for name in FILES:
        url = f'https://raw.githubusercontent.com/aristoteleo/veckit/{commit}/{name}'
        body = get(url)
        path = DEST/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        records.append({'path': name, 'url': url, 'sha256': hashlib.sha256(body).hexdigest()})
    manifest.write_text(json.dumps({'repository': 'https://github.com/aristoteleo/veckit',
        'commit': commit, 'files': records, 'biological_data_downloaded': False}, indent=2), encoding='utf-8')
    print(json.dumps({'commit': commit, 'files': len(records), 'biological_data_downloaded': False}))


if __name__ == '__main__': main()
