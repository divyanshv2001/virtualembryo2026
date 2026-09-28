"""Retrieve the public WT early-gastrulation research atlas, never the KO files."""
import hashlib
import json
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from datetime import datetime, timezone

DEST = Path(__file__).resolve().parent / 'private' / 'early_atlas'
BASE = 'https://gastrulation.stemcells.cam.ac.uk/'


class Links(HTMLParser):
    def __init__(self): super().__init__(); self.links = []
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.links.extend(v for k, v in attrs if k == 'href')


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(BASE, headers={'User-Agent': 'VirtualEmbryoResearch/1.0'})
    html = urllib.request.urlopen(request, timeout=30).read().decode()
    parser = Links(); parser.feed(html)
    print(json.dumps({'public_links': parser.links}))
    (DEST / 'source_page.html').write_text(html, encoding='utf-8')
    records = []
    for name in ['metadata.txt', 'counts.gz']:
        url = BASE + 'data/' + name
        path = DEST / name
        if not path.exists():
            request = urllib.request.Request(url, headers={'User-Agent': 'VirtualEmbryoResearch/1.0'})
            with urllib.request.urlopen(request, timeout=30) as response, path.with_suffix(path.suffix+'.part').open('wb') as f:
                total = 0
                while block := response.read(1024*1024):
                    total += len(block)
                    if total > 300*1024*1024: raise ValueError('Unexpected download size')
                    f.write(block)
            path.with_suffix(path.suffix+'.part').replace(path)
        records.append({'file': name, 'url': url, 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {'retrieved_utc': datetime.now(timezone.utc).isoformat(),
        'source': BASE, 'paper': 'https://doi.org/10.1038/nature18633',
        'declared_stages': [6.5, 7.0, 7.5, 7.75], 'genotype': 'WT atlas; separate Tal1 files excluded',
        'task_scope': 'T1 local early-stage research backtests only; not T2',
        'usage': 'Public author-released research data; no redistribution or use in a competition prediction in this run.',
        'transfer_limit': 'SMART-seq/FACS early mesoderm differs from supplied later-stage heart RNA; local scores are not T1:val scores.',
        'files': records}
    (DEST/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'downloaded': records}))


if __name__ == '__main__': main()
