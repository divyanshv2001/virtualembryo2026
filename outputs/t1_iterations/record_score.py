"""Record explicit user-supplied score feedback with an artifact checksum."""
import argparse
import json
import math
import sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'outputs/t1_run'))
from run_t1 import digest

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--round',default='round_01');p.add_argument('--artifact',required=True)
    p.add_argument('--score',type=float,required=True);p.add_argument('--metrics-json',type=Path)
    args=p.parse_args()
    if not args.round.replace('_','').isalnum():raise ValueError('Invalid round')
    artifact=Path(args.artifact).resolve()
    if not artifact.is_relative_to(ROOT) or artifact.suffix!='.h5ad':raise ValueError('Artifact must be a workspace prediction')
    if not math.isfinite(args.score) or not 0<=args.score<=100:raise ValueError('Invalid score')
    metrics=json.loads(args.metrics_json.read_text()) if args.metrics_json else None
    if metrics is not None:
        if not isinstance(metrics,dict):raise ValueError('Metrics must be an object')
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in metrics.values()):
            raise ValueError('Invalid component values')
    path=ROOT/'outputs/t1_iterations/private'/args.round/'score_ledger.json'
    ledger=json.loads(path.read_text())
    observation={'artifact':artifact.relative_to(ROOT).as_posix(),'sha256':digest(artifact),'score':args.score,
                 'components':metrics,'source':'Explicit user-supplied result','recorded_utc':datetime.now(timezone.utc).isoformat(),
                 'verification':'not fetched from authenticated portal','board':'T1:val'}
    duplicate=any(r['sha256']==observation['sha256'] and r['score']==args.score and r['components']==metrics for r in ledger['observations'])
    if not duplicate:
        ledger['observations'].append(observation)
        path.write_text(json.dumps(ledger,indent=2),encoding='utf-8')
    print(json.dumps({'new_record_added':not duplicate,'observations':len(ledger['observations']),
                      'best_reported_score':max(r['score'] for r in ledger['observations'])}))

if __name__=='__main__':main()
