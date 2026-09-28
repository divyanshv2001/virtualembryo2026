"""Controlled training-only T1 candidates; actual local batch events recorded.

Scores are recorded for selection only, never inverted to estimate target data.
No API, portal login, upload, official scorer or Agent Team eligibility claim.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import anndata as ad
import numpy as np
from scipy import sparse

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
T1=ROOT/'outputs/t1_run'
sys.path.insert(0,str(T1))
from run_t1 import digest,means,validate

def now():return datetime.now(timezone.utc).isoformat()
def change(block,delta,alpha,preserve_zero=False):
    original=np.asarray(block,dtype=np.float32)
    before=original.astype(np.float64)+alpha*np.asarray(delta,dtype=np.float64)
    if preserve_zero:before=np.where(original>0,before,0)
    negative=int((before<0).sum())
    result=np.maximum(before,0).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('Nonfinite candidate')
    return result,negative

def append_event(path,kind,**data):
    event={'timestamp_utc':now(),'event':kind,**data}
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(event)+'\n');f.flush()
    print(json.dumps({'event':kind,**data}),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--round',default='round_01')
    args=parser.parse_args()
    if not args.round.replace('_','').isalnum():raise ValueError('Invalid round name')
    out=HERE/'private'/args.round
    if out.exists():raise SystemExit('Round already exists; preserve its artifacts. Choose a new round name.')
    out.mkdir(parents=True)
    eventlog=out/'batch_events.jsonl'
    report=json.loads((T1/'run_report.json').read_text())
    donors=np.load(T1/'donor_rows.npy',allow_pickle=False)
    panel=(T1/'T1__val.genes.txt').read_text().splitlines()
    spec=json.loads((T1/'index.json').read_text())['T1:val']
    configs=[{'name':'shrunk_shift_a010','alpha':.1,'preserve_zero':False},
             {'name':'shrunk_shift_a025','alpha':.25,'preserve_zero':False},
             {'name':'zero_preserving_shift_a025','alpha':.25,'preserve_zero':True}]
    plan={'created_before_candidate_execution_utc':now(),'board':'T1:val','target':'E10.5',
          'methods':configs,'donor_count':len(donors),'donor_rows_sha256':digest(T1/'donor_rows.npy'),
          'seed':report['seed'],'input_files':report['source_files'],'code_sha256':digest(Path(__file__)),
          'prompt':'Develop controlled T1 candidates after the user reported the full mean-shift score as 48.1.',
          'selection_policy':'Evaluate persistence control then individual candidates; returned scores select candidates only. No target-property recovery.',
          'scientific_status':'Development batch, no locally validated predictive performance.',
          'agent_team_configuration_lock':False,'note':'A fixed candidate batch is not a full locked autonomous agent run.'}
    (out/'batch_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    append_event(eventlog,'plan_frozen',plan_sha256=digest(out/'batch_plan.json'),candidates=[c['name'] for c in configs])
    ledger={'observations':[{'artifact':'outputs/t1_run/T1_val__pseudobulk_shift_exploratory.h5ad',
              'sha256':report['artifacts'][1]['validation']['sha256'],'score':48.1,'components':None,
              'source':'User-provided submissions screenshot; scored 2026-09-28 08:01 as displayed; timezone unspecified',
              'verification':'user-reported screenshot; no authenticated portal query','board':'T1:val'}],
              'remaining_t1_daily_scored_attempts_at_screenshot':7,'live_jev_requests':0}
    (out/'score_ledger.json').write_text(json.dumps(ledger,indent=2),encoding='utf-8')
    files=[ROOT/'data/E8.5_RNA.h5ad',ROOT/'data/E9.5_RNA.h5ad']
    stages=[]
    try:
        for path,expected in zip(files,report['source_files']):
            if digest(path)!=expected['sha256']:raise ValueError('Training file changed since previous run')
            stages.append(ad.read_h5ad(path,backed='r'))
        if len(set(donors.tolist()))!=len(donors) or donors.min()<0 or donors.max()>=stages[1].n_obs:
            raise ValueError('Invalid donor ledger')
        column_indices=[a.var_names.get_indexer(panel) for a in stages]
        if any((c<0).any() for c in column_indices):raise ValueError('Missing official panel gene')
        baseline=ad.read_h5ad(T1/'T1_val__sampled_copy_last.h5ad')
        if digest(T1/'T1_val__sampled_copy_last.h5ad')!=report['artifacts'][0]['validation']['sha256']:
            raise ValueError('Baseline artifact changed')
        base=baseline.X.toarray() if sparse.issparse(baseline.X) else np.asarray(baseline.X)
        append_event(eventlog,'input_verification_passed',baseline_sha256=digest(T1/'T1_val__sampled_copy_last.h5ad'))
        append_event(eventlog,'training_means_started',stages=['E8.5','E9.5'],annotation_match='exact strings only')
        labels_a,mu_a=means(stages[0],column_indices[0])
        labels_b,mu_b=means(stages[1],column_indices[1])
        shared=set(mu_a)&set(mu_b)
        selected=labels_b[donors]
        append_event(eventlog,'training_means_completed',shared_labels=len(shared),last_only_labels=len(set(mu_b)-shared))
        records=[]
        baseline_zeros=int((base==0).sum())
        for config in configs:
            name=config['name']
            append_event(eventlog,'candidate_started',name=name,configuration=config)
            result=base.copy();negative=0;changed_rows=0
            for label in np.unique(selected):
                if label not in shared:continue
                mask=selected==label
                result[mask],clipped=change(base[mask],mu_b[label]-mu_a[label],config['alpha'],config['preserve_zero'])
                negative+=clipped;changed_rows+=int(mask.sum())
            path=out/f'T1_val__{name}.h5ad'
            candidate=ad.AnnData(X=sparse.csr_matrix(result),obs=baseline.obs.copy(),var=baseline.var.copy())
            candidate.write_h5ad(path,compression='gzip')
            checks=validate(path,panel,spec)
            reload=ad.read_h5ad(path)
            diff=reload.X-candidate.X
            if diff.nnz and np.any(diff.data!=0):raise ValueError('Candidate save/reload changed values')
            diagnostic={'negative_entries_clipped':negative,'new_nonzero_entries':int(((base==0)&(result>0)).sum()),
                        'nonzero_to_zero_entries':int(((base>0)&(result==0)).sum()),'baseline_zero_entries':baseline_zeros,
                        'mean_absolute_change':float(np.mean(np.abs(result-base),dtype=np.float64)),
                        'changed_rows':changed_rows,'scope':'Training/output diagnostics, not future performance estimates'}
            if config['preserve_zero'] and diagnostic['new_nonzero_entries']:
                raise ValueError('Zero-preservation violated')
            records.append({'method':name,'configuration':config,'artifact':path.relative_to(ROOT).as_posix(),
                            'validation':checks,'diagnostics':diagnostic,'official_score':None,'uploaded':False})
            append_event(eventlog,'candidate_validated',name=name,sha256=checks['sha256'],diagnostics=diagnostic)
        summary={'plan_sha256':digest(out/'batch_plan.json'),'candidates':records,
                 'next_submission':'outputs/t1_run/T1_val__sampled_copy_last.h5ad',
                 'next_reason':'Unscored matched persistence control separates extrapolation harm from submission sampling.',
                 'followup':'Choose one smaller-shift candidate after component score feedback; never infer held-out values.',
                 'no_local_future_target':True,'official_performance_unmeasured_for_new_candidates':True,
                 'live_jev_requests':0,'agent_team_eligibility':'unresolved development session; no full locked run certified'}
        (out/'iteration_report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        append_event(eventlog,'batch_completed',validated_candidates=len(records),next_submission=summary['next_submission'])
    except Exception as exc:
        append_event(eventlog,'batch_failed',exception_type=type(exc).__name__)
        raise
    finally:
        for a in stages:a.file.close()

if __name__=='__main__':main()
