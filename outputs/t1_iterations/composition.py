"""Training-only capped composition trends; empirical expression is unchanged."""
import argparse
import json
import sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
T1=ROOT/'outputs/t1_run'
sys.path.insert(0,str(T1))
from run_t1 import digest,validate
from iterate import append_event,now

def forecast_weights(labels_a,labels_b,strength=.5,cap=2):
    if not np.isfinite(strength) or not 0<=strength<=1 or not np.isfinite(cap) or cap<1:
        raise ValueError('Invalid composition parameters')
    types,nb=np.unique(labels_b,return_counts=True)
    ta,na=np.unique(labels_a,return_counts=True)
    pa={k:n/len(labels_a) for k,n in zip(ta,na)}
    pb=nb/len(labels_b)
    multiplier=np.ones(len(types),dtype=np.float64)
    for i,label in enumerate(types):
        if label in pa:
            multiplier[i]=np.clip((pb[i]/pa[label])**strength,1/cap,cap)
    weights=pb*multiplier;weights/=weights.sum()
    return types,weights,multiplier

def allocate(weights,cells):
    expected=np.asarray(weights)*cells
    counts=np.floor(expected).astype(int)
    remain=cells-int(counts.sum())
    counts[np.argsort(-(expected-counts),kind='stable')[:remain]]+=1
    if counts.sum()!=cells or (counts<0).any():raise ValueError('Invalid quotas')
    return counts

def main():
    p=argparse.ArgumentParser();p.add_argument('--round',default='round_02_composition');p.add_argument('--cells',type=int,default=2000)
    args=p.parse_args()
    if not args.round.replace('_','').isalnum():raise ValueError('Invalid round')
    out=HERE/'private'/args.round
    if out.exists():raise SystemExit('Round exists; preserve original artifacts and use a new name.')
    spec=json.loads((T1/'index.json').read_text())['T1:val']
    if not spec['min_cells']<=args.cells<=spec['max_cells']:raise ValueError('Cell count outside board bounds')
    out.mkdir(parents=True)
    events=out/'batch_events.jsonl'
    configs=[{'name':'stratified_persistence','strength':0.,'cap':2},
             {'name':'composition_trend_a050','strength':.5,'cap':2}]
    report=json.loads((T1/'run_report.json').read_text())
    plan={'created_before_execution_utc':now(),'board':'T1:val','methods':configs,'cells':args.cells,'seed':20260928,
          'input_files':report['source_files'],'code_sha256':digest(Path(__file__)),
          'formula':'pi_10(k) proportional to p_9(k) * clip((p_9(k)/p_8(k))^strength,1/cap,cap) for exact shared labels; unmatched last-stage label multiplier=1.',
          'selection':'Candidates selected only after user-provided official results; no target-property inversion.',
          'scope':'Observed sampling/annotation composition forecast, not identified abundance or guaranteed biological change.',
          'agent_team_configuration_lock':False}
    (out/'batch_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    append_event(events,'composition_plan_frozen',sha256=digest(out/'batch_plan.json'))
    panel=(T1/'T1__val.genes.txt').read_text().splitlines()
    canonical=__import__('hashlib').sha256('\n'.join(panel).encode()).hexdigest()
    if canonical[:len(spec['genes_sha256'])]!=spec['genes_sha256']:raise ValueError('Official panel hash mismatch')
    stages=[]
    try:
        for source in report['source_files']:
            path=ROOT/source['path']
            if digest(path)!=source['sha256']:raise ValueError('Source changed')
            stages.append(ad.read_h5ad(path,backed='r'))
        labels=[]
        for a in stages:
            v=a.obs['celltype'].astype('string')
            if v.isna().any() or (v=='').any():raise ValueError('Missing annotation')
            labels.append(v.to_numpy(dtype=str))
        last=stages[1];columns=last.var_names.get_indexer(panel)
        if (columns<0).any():raise ValueError('Panel missing from source')
        rngseed=plan['seed'];records=[]
        for config in configs:
            append_event(events,'candidate_started',name=config['name'])
            types,weights,mults=forecast_weights(*labels,config['strength'],config['cap'])
            counts=allocate(weights,args.cells)
            rng=np.random.Generator(np.random.PCG64(rngseed))
            chosen=[]
            for label,count in zip(types,counts):
                pool=np.flatnonzero(labels[1]==label)
                if count>len(pool):raise ValueError('Quota exceeds available empirical cells; no duplicates silently introduced')
                chosen.extend(rng.choice(pool,count,replace=False).tolist())
            rows=np.sort(np.asarray(chosen,dtype=np.int64))
            if len(set(rows.tolist()))!=args.cells:raise ValueError('Duplicate donor row')
            source=last.X[rows,:][:,columns].astype(np.float32)
            x=source.tocsr() if sparse.issparse(source) else sparse.csr_matrix(source)
            artifact=out/f'T1_val__{config["name"]}.h5ad'
            a=ad.AnnData(X=x,obs=pd.DataFrame(index=[f'prediction_{i:05}' for i in range(args.cells)]),var=pd.DataFrame(index=panel))
            a.write_h5ad(artifact,compression='gzip')
            check=validate(artifact,panel,spec)
            reloaded=ad.read_h5ad(artifact);difference=reloaded.X-x
            if difference.nnz and np.any(difference.data):raise ValueError('Expression changed during export')
            check['source_expression_unchanged']=True
            np.save(out/f'{config["name"]}_donors.npy',rows)
            table=[{'training_label':str(k),'forecast_weight':float(w),'trend_multiplier':float(m),'output_count':int(n)} for k,w,m,n in zip(types,weights,mults,counts)]
            record={'method':config['name'],'configuration':config,'artifact':artifact.relative_to(ROOT).as_posix(),
                    'validation':check,'local_composition_table':table,'official_score':None,'uploaded':False}
            records.append(record)
            append_event(events,'candidate_validated',name=config['name'],sha256=check['sha256'],source_expression_unchanged=True)
        summary={'plan_sha256':digest(out/'batch_plan.json'),'candidates':records,'live_jev_requests':0,
                 'prediction_scope':'Change empirical cell-state weights only; no expression shift or held-out readout.',
                 'limitations':['Exact labels not harmonized','Dissection/capture can mimic trends','Unmatched label trends unidentified','Future-state support missing','No future target test locally'],
                 'next_test':'Compare stratified persistence and composition trend; do not claim score superiority until official results.'}
        (out/'iteration_report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        append_event(events,'batch_completed',validated_candidates=len(records))
    finally:
        for a in stages:a.file.close()

if __name__=='__main__':main()
