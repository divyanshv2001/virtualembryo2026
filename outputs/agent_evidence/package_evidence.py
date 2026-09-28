"""Package recorded visible events, not a retrospectively invented trajectory.

Only this workspace/root session and its delegated sessions are read. Framework
reasoning, privileged instructions and credential values are excluded explicitly.
Archives stay ignored and local. No upload or new model execution occurs.
"""
import hashlib
import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SESSIONS=Path('C:/Users/vishw/.codex/sessions')
ROOT_ID='01a0e36c-d539-7a83-b6c5-a6af374a680b'
PRIVATE=HERE/'private'
KEY=ROOT/'outputs/research_workflow/.secrets/typesafe.key'
def sha(data):return hashlib.sha256(data).hexdigest()
def read_secret():
    if not KEY.exists():return ''
    value=KEY.read_text(encoding='utf-8-sig').strip()
    if value.startswith('TYPESAFE_API_KEY='):value=value.split('=',1)[1].strip()
    return value.strip('\"\'')

PATTERN=re.compile(r'apikey(?:\\)?_[0-9a-fA-F]{32}(?:\\)?_[0-9a-fA-F]{64}')
def redact(value,secret,counter):
    if isinstance(value,str):
        if secret and secret in value:
            counter['exact_credential_replacements']+=value.count(secret)
            value=value.replace(secret,'[REDACTED_AUTHENTICATION_CREDENTIAL]')
        value,n=PATTERN.subn('[REDACTED_AUTHENTICATION_CREDENTIAL]',value)
        counter['credential_pattern_replacements']+=n
        return value
    if isinstance(value,list):return [redact(v,secret,counter) for v in value]
    if isinstance(value,dict):return {k:redact(v,secret,counter) for k,v in value.items()}
    return value

def allowed(record):
    kind=record.get('type');p=record.get('payload',{})
    if kind=='response_item':
        t=p.get('type')
        if t=='message':
            # User prompts come from the framework's visible user_message events.
            return p.get('role')=='assistant' and p.get('channel') not in ('analysis','summary')
        return t in ('function_call','function_call_output','custom_tool_call','custom_tool_call_output')
    if kind=='event_msg':
        return p.get('type') in ('task_started','task_complete','user_message','agent_message','web_search_end','patch_apply_end','item_completed') and p.get('channel')!='analysis'
    return False

def put(z,name,data,secret):
    if isinstance(data,str):data=data.encode('utf-8')
    # Fail closed rather than allow undisclosed credential leakage into an archive.
    if secret and secret.encode() in data:
        raise ValueError('Credential in archive input; value withheld')
    if PATTERN.search(data.decode('utf-8',errors='ignore')):
        raise ValueError('Credential pattern in archive input; value withheld')
    z.writestr(name,data)

def main():
    PRIVATE.mkdir(parents=True,exist_ok=True)
    secret=read_secret()
    sessions=[]
    for path in sorted(SESSIONS.rglob('*.jsonl')):
        raw=path.read_bytes()
        lines=raw.decode('utf-8').splitlines()
        if not lines:continue
        first=json.loads(lines[0]);meta=first.get('payload',{})
        source=meta.get('source');spawn=source.get('subagent',{}).get('thread_spawn',{}) if isinstance(source,dict) else {}
        if meta.get('id')!=ROOT_ID and spawn.get('parent_thread_id')!=ROOT_ID:continue
        if Path(meta.get('cwd','')).resolve()!=ROOT.resolve():
            raise ValueError('Session workspace mismatch')
        records=[json.loads(line) for line in lines if line.strip()]
        sessions.append((path,raw,meta,records))
    root=next((row for row in sessions if row[2].get('id')==ROOT_ID),None)
    if root is None:raise ValueError('Original root session missing')
    command='run_t1.py --cells 2000 --seed 20260928 --shift-candidate'
    if not any(command in json.dumps(r) for r in root[3]):
        raise ValueError('T1 execution command absent from original log')
    models=sorted({r.get('payload',{}).get('model') for _,_,_,rs in sessions for r in rs if r.get('type')=='turn_context' and r.get('payload',{}).get('model')})
    provenance={'exported_at_utc':datetime.now(timezone.utc).isoformat(),'framework':'Codex; source metadata vscode',
                'recorded_model_strings':models,'root_session_id':ROOT_ID,'t1_execution_command_recorded':True,
                'configuration_lock_recorded':False,'eligibility':'unresolved; development-run evidence, not a certified Agent Team entry',
                'scope':'Recorded externally visible event export. Original framework reasoning, privileged system/developer instructions, world state, compaction internals and account/guardian sessions excluded. Credentials redacted. No events, timestamps or configuration lock invented. Full source hashes refer to local originals, not public exports.',
                'source_sessions':[],'archives':[]}
    trajectory=PRIVATE/'trajectory.zip';prompts=PRIVATE/'prompts.zip';harness=PRIVATE/'harness.zip'
    with zipfile.ZipFile(trajectory,'w',zipfile.ZIP_DEFLATED) as tz,zipfile.ZipFile(prompts,'w',zipfile.ZIP_DEFLATED) as pz:
        for path,raw,meta,records in sessions:
            count=Counter();omitted=Counter();kept=[];prompt_rows=[]
            for record in records:
                if allowed(record):
                    clean=redact(record,secret,count);kept.append(json.dumps(clean,ensure_ascii=False))
                    if record.get('type')=='event_msg' and record.get('payload',{}).get('type')=='user_message':prompt_rows.append(clean)
                else:
                    omitted[record.get('type','unknown')+':'+str(record.get('payload',{}).get('type',''))]+=1
            record={'id':meta['id'],'source_file':path.name,'original_sha256':sha(raw),'original_bytes':len(raw),
                    'original_records':len(records),'visible_records_exported':len(kept),'visible_prompt_records':len(prompt_rows),
                    'omitted_record_categories':dict(omitted),'redactions':dict(count)}
            provenance['source_sessions'].append(record)
            put(tz,'recorded_events/'+path.name,'\n'.join(kept)+'\n',secret)
            put(pz,'recorded_prompts/'+meta['id']+'.jsonl','\n'.join(json.dumps(r,ensure_ascii=False) for r in prompt_rows)+'\n',secret)
        # This document was actually read as the adopted task source; it is not a new starting prompt.
        put(pz,'adopted_source_document/agenticprompt',(ROOT/'Neurl IPS 2026/agenticprompt').read_bytes(),secret)
        text='Recorded-event exports from the original Codex sessions. Source event order/timestamps retained. Credentials replaced in place; omissions are itemized in export_provenance.json. No retrospective summary is represented as a trajectory. This interactive development session has no recorded pre-run configuration lock. A valid autonomous-track entry may require a new properly locked run; organizer acceptance is unresolved. Hidden framework reasoning/instructions are not included. The prompt archive contains recorded visible user/delegation prompts, not all platform-controlled instructions.\n'
        for z in (tz,pz):
            put(z,'READ_ME.txt',text,secret)
            put(z,'export_provenance.json',json.dumps(provenance,indent=2),secret)
    with zipfile.ZipFile(harness,'w',zipfile.ZIP_DEFLATED) as hz:
        tools=ROOT/'outputs/t1_run'
        names=['run_t1.py','inspect_inputs.py','fetch_contract.py','test_t1.py','requirements-lock.txt','README.md','T1__val.genes.txt','index.json','contract_sources.json']
        sourcehashes=[]
        for name in names:
            data=(tools/name).read_bytes();put(hz,'t1_task_tools/'+name,data,secret)
            sourcehashes.append({'path':name,'sha256':sha(data)})
        # Contemporaneous generated manifest is supporting evidence, not trajectory.
        report=tools/'run_report.json'
        if report.exists():put(hz,'supporting_run_report.json',report.read_bytes(),secret)
        put(hz,'source_hashes.json',json.dumps(sourcehashes,indent=2),secret)
        put(hz,'READ_ME.txt','Task tool code called by Codex in the recorded T1 run. Codex itself is the agent framework; these files are not a custom agent episode runner, local official scorer or uploader. This archive is supporting harness/tool material, not an eligibility certification. No biological matrices, donor rows, credentials or framework installation are included.\n',secret)
    for path in (trajectory,prompts,harness):
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None:raise ValueError('Invalid ZIP')
        if path.stat().st_size>200_000_000:raise ValueError('Evidence exceeds individual-file limit')
        provenance['archives'].append({'file':path.name,'bytes':path.stat().st_size,'sha256':sha(path.read_bytes())})
    if sum(r['bytes'] for r in provenance['archives'])>600_000_000:raise ValueError('Evidence exceeds total-team limit')
    (PRIVATE/'export_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps({'archives':provenance['archives'],'source_sessions':len(sessions),'model_strings':models,'configuration_lock_recorded':False,'uploaded':False,'eligibility':'unresolved'},indent=2))

if __name__=='__main__':main()
