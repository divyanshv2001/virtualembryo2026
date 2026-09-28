"""Inspect/download the author-linked E6.5-E9.5 atlas; never submit files."""
import argparse
import json
import urllib.request
import urllib.error
import hashlib
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = HERE/'private/extended_atlas'
SHARE = 'https://cloud.mrc-lmb.cam.ac.uk/s/yxq7FRtYsLyF3jQ'
MIRROR = 'https://cells-test.gi.ucsc.edu/ext-mouse-atlas/'


def download(name):
    path = DEST/name
    manifest_path = DEST/(name+'.download.json')
    if path.exists() and manifest_path.exists():
        record = json.loads(manifest_path.read_text())
        print(json.dumps({'event':'cached','file':name,'bytes':path.stat().st_size}),flush=True)
        return record
    part = DEST/(name+'.part')
    # Existing partials are deliberately restarted; they have no pinned response manifest.
    request = urllib.request.Request(MIRROR+name,headers={'User-Agent':'VirtualEmbryoResearch/1.0'})
    sha = hashlib.sha256(); started = time.monotonic(); last = started; total = 0
    with urllib.request.urlopen(request,timeout=45) as response, part.open('wb') as f:
        expected = int(response.headers.get('Content-Length',0))
        etag = response.headers.get('ETag')
        while block := response.read(4*1024*1024):
            f.write(block); sha.update(block); total += len(block)
            if time.monotonic()-last>25:
                print(json.dumps({'event':'download_progress','file':name,'bytes':total,'expected':expected,
                    'megabytes_per_second':round(total/(time.monotonic()-started)/1e6,2)}),flush=True)
                last = time.monotonic()
    if expected and total!=expected: raise ValueError('Incomplete download')
    part.replace(path)
    record = {'url':MIRROR+name,'file':name,'bytes':total,'sha256':sha.hexdigest(),'etag':etag}
    manifest_path.write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps({'event':'download_complete',**record}),flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    if args.download:
        records = [download(name) for name in ['dataset.json','exprMatrix.json','meta.tsv','exprMatrix.bin']]
        manifest = {'source_page':'https://marionilab.github.io/ExtendedMouseAtlas/',
            'publication':'https://doi.org/10.1242/dev.201867',
            'publication_license':'CC BY 4.0', 'author_linked_mirror':MIRROR,
            'declared_stages':'E6.5 through E9.5; actual metadata gate required before training',
            'files':records,'provided_embeddings_used':False,'submissions_used':0}
        (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        return
    urls = [SHARE,
        SHARE+'/download?path=%2F&files=embryo_raw_counts.h5ad',
        'https://www.ebi.ac.uk/biostudies/api/v1/studies/E-MTAB-11763']
    records = []
    for i,url in enumerate(urls):
        try:
            request = urllib.request.Request(url,headers={'User-Agent':'VirtualEmbryoResearch/1.0'})
            with urllib.request.urlopen(request,timeout=25) as response:
                content_type = response.headers.get('Content-Type','')
                if i==1 and ('octet-stream' in content_type or 'hdf' in content_type):
                    records.append({'url':url,'status':response.status,'content_type':content_type,
                        'bytes':response.headers.get('Content-Length'),'data_endpoint_available':True})
                else:
                    body = response.read(2*1024*1024)
                    suffix = '.json' if 'json' in content_type else '.html'
                    (DEST/f'probe_{i}{suffix}').write_bytes(body)
                    records.append({'url':url,'status':response.status,'content_type':content_type,'metadata_bytes':len(body)})
        except Exception as e:
            records.append({'url':url,'error_type':type(e).__name__,'error':str(e)})
    (DEST/'access_probe.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps(records,indent=2))


if __name__=='__main__': main()
