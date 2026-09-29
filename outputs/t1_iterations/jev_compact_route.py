"""One-attempt compact advisory routing with validated cache and safe accounting."""
import argparse,hashlib,json,sys,urllib.request,urllib.error
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'submission_preparation'))
from compact_jev import canonical,validate_answers,cache_key


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--packet',required=True);parser.add_argument('--record',required=True);args=parser.parse_args()
    packet=json.loads(Path(args.packet).read_text());out=Path(args.record)
    fingerprint=cache_key(packet)
    if out.exists():
        record=json.loads(out.read_text())
        if record.get('semantic_cache_key')==fingerprint and record.get('status')=='validated':
            print(json.dumps({'status':'cache_hit','new_requests':0,'answers':record['answers'],'usage':record['usage']}));return
        raise ValueError('Prior attempt retained; do not silently repeat a paid request')
    raw=canonical(packet).encode()
    if len(raw)>3000:raise ValueError('Compact3000byte budget exceeded')
    root=Path(__file__).resolve().parents[2]
    key=(root/'outputs/research_workflow/.secrets/typesafe.key').read_text(encoding='utf-8-sig').strip()
    if key.startswith('TYPESAFE_API_KEY='):key=key.split('=',1)[1].strip()
    if len(key)>1 and key[0] in ['"',"'"] and key[-1]==key[0]:key=key[1:-1]
    if not key or '\n' in key:raise ValueError('Invalid local key format; value not displayed')
    record={'status':'attempt_started','semantic_cache_key':fingerprint,'attempts':1,'payload_bytes':len(raw),'input':packet,'usage':None,'scope':'Advisory routing only, no score certification or token savings guarantee.'}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(record,indent=2))
    request=urllib.request.Request('https://api.typesafe.ai/v1/systemone',data=raw,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
    try:
        with urllib.request.urlopen(request,timeout=30) as response:body=json.load(response)
        answers=validate_answers(packet['questions'],body.get('answers',{}))
        safe={k:{field:value for field,value in a.items() if field in ['choice','confidence','noul']} for k,a in answers.items()}
        usage={k:body.get('usage',{}).get(k) for k in ['input_tokens','output_tokens']}
        model=body.get('model')
        if not isinstance(model,str) or key in model:raise ValueError('Invalid model metadata')
        record.update(status='validated',answers=safe,usage=usage,returned_model=model)
    except urllib.error.HTTPError as exc:record.update(status='request_failed',http_status=exc.code)
    except (urllib.error.URLError,TimeoutError):record.update(status='request_unavailable')
    except (ValueError,TypeError,KeyError):record.update(status='invalid_response')
    out.write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k not in ['input','semantic_cache_key']}))


if __name__=='__main__':main()
