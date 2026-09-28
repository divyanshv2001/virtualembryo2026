"""Download public T1 panel/index only; no authentication or biological data."""
import hashlib
import json
import urllib.request
from pathlib import Path
here=Path(__file__).resolve().parent
sources={'T1__val.genes.txt':'https://virtualembryo.ai/challenge/panels/T1__val.genes.txt','index.json':'https://virtualembryo.ai/challenge/panels/index.json'}
records=[]
for name,url in sources.items():
    path=here/name
    if not path.exists():
        request=urllib.request.Request(url,headers={'User-Agent':'VirtualEmbryoResearch/1.0','Accept':'text/plain,application/json'})
        with urllib.request.urlopen(request,timeout=30) as response:
            body=response.read()
        if name.endswith('.json'):
            json.loads(body)
        elif len(body.decode().splitlines())!=32285:
            raise ValueError('Unexpected official panel length')
        path.write_bytes(body)
    records.append({'path':name,'url':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(here/'contract_sources.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print(json.dumps(records,indent=2))
