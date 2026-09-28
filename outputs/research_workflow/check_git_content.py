"""Scan staged blobs without printing secret values; no secret is persisted."""
import json
import re
import subprocess
from pathlib import Path
here = Path(__file__).resolve().parent
root = here.parents[1]
git = ['git','-c','safe.directory='+root.as_posix()]
def run(*args):
    return subprocess.check_output(git+list(args),cwd=root)
paths = [p.decode('utf-8') for p in run('diff','--cached','--name-only','-z').split(b'\0') if p]
secret = (here/'.secrets/typesafe.key').read_text(encoding='utf-8-sig').strip()
if secret.startswith('TYPESAFE_API_KEY='):
    secret=secret.split('=',1)[1].strip().strip('\"\'')
errors=[]
for path in paths:
    if any(part in ('.secrets','.venv','__pycache__','.pytest_cache') for part in Path(path).parts):
        errors.append({'path':path,'reason':'excluded directory staged'})
        continue
    blob=run('show',':'+path)
    if secret and secret.encode() in blob:
        errors.append({'path':path,'reason':'replacement credential found; value withheld'})
    if re.search(rb'apikey_[0-9a-fA-F]{32}_[0-9a-fA-F]{64}',blob):
        errors.append({'path':path,'reason':'credential pattern found; value withheld'})
    if re.search(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----\s+[A-Za-z0-9+/=\r\n]{32,}\s+-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',blob):
        errors.append({'path':path,'reason':'private key found; value withheld'})
print(json.dumps({'staged_files_scanned':len(paths),'errors':errors,'secret_values_printed':False},indent=2))
raise SystemExit(bool(errors) or not paths)
