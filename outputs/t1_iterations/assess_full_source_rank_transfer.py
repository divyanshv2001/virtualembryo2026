"""Screen full-source past trend ranks on historical stage holdouts.

This is a pseudobulk gene-rank diagnostic, not the challenge's full scorer.
"""
import json
import numpy as np
import pandas as pd
from collections import Counter

from train_extended_atlas import HERE
from cohort_groups import GROUPS
from offline_backtest import load_core
from run_t1 import digest
from iterate import now


STAGES=np.array([7.5,7.75,8.,8.25,8.5])


def signed_score(core,predicted,truth,reference,mapped):
    delta=truth-reference
    up=np.flatnonzero(mapped & (delta>=.25))
    down=np.flatnonzero(mapped & (delta<=-.25))
    if len(up)+len(down)<5:return {'valid':False,'true_up':len(up),'true_down':len(down)}
    chance=max(core._signed_overlap(reference,up,down)[0],core._signed_overlap(-reference,up,down)[0])
    overlap=core._signed_overlap(predicted,up,down)[0]
    return {'valid':True,'true_up':len(up),'true_down':len(down),
            'signed_overlap':float(overlap),'chance':float(chance),
            'chance_adjusted_score':float((overlap-chance)/(1-chance)) if chance<1 else None}


def trend(means,cutoff_index):
    a,b,c=means[cutoff_index-2:cutoff_index+1]
    left=b-a;right=c-b
    return np.where(left*right>0,2*(left+right),0.)


def main():
    source=HERE/'private/full_atlas_stage_means_01'
    report=json.loads((source/'report.json').read_text())
    if report.get('status')!='completed':raise ValueError('Require complete full-source sufficient statistics')
    with np.load(source/'stage_means.npz') as saved:
        full={name:saved[name].astype(float) for name in ['whole','cardiac']}
        mapped=saved['mapped'].astype(bool)
        np.testing.assert_array_equal(saved['stages'],STAGES)
        counts={name:saved[name+'_counts'] for name in ['whole','cardiac']}
    prepared=HERE/'private/associated_prepared_01'
    x=np.load(prepared/'expression.npy',mmap_mode='r')
    meta=pd.read_csv(prepared/'selected_metadata.csv')
    symbols=pd.read_csv(prepared/'genes.csv').symbol.fillna('').tolist()
    panel=(HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
    c=Counter(symbols);lookup={s:i for i,s in enumerate(symbols) if s and c[s]==1}
    official=np.array([i for i,s in enumerate(panel) if s in lookup]);atlas=np.array([lookup[panel[i]] for i in official])
    sample={};sample_count={}
    for group in ['whole','cardiac']:
        label=meta.celltype_extended_atlas.isin(GROUPS['cardiac']).to_numpy() if group=='cardiac' else np.ones(len(meta),bool)
        means=np.zeros((len(STAGES),len(panel)),dtype=float);n=[]
        for t,stage in enumerate(STAGES):
            rows=np.flatnonzero((meta.numeric_stage.to_numpy(float)==stage)&label)
            n.append(len(rows))
            if len(rows):
                for start in range(0,len(official),256):
                    sl=slice(start,min(start+256,len(official)))
                    means[t,official[sl]]=np.asarray(x[np.ix_(rows,atlas[sl])],float).mean(0)
        sample[group]=means;sample_count[group]=np.array(n)
    core,_=load_core();results=[]
    for group in ['whole','cardiac']:
        for cutoff_index in [2,3]:
            target_index=cutoff_index+1
            n_full=int(counts[group][target_index]);n_sample=int(sample_count[group][target_index])
            if n_full<=n_sample:raise ValueError('No held-out source target cells')
            # Sampled cohort is a subset of full source; remove those target cells.
            truth=(n_full*full[group][target_index]-n_sample*sample[group][target_index])/(n_full-n_sample)
            base=full[group][cutoff_index]
            controls={'sampled_past_trend':trend(sample[group],cutoff_index),
                      'full_past_trend':trend(full[group],cutoff_index)}
            results.append({'group':group,'cutoff':float(STAGES[cutoff_index]),
                'target':float(STAGES[target_index]),'full_past_counts':counts[group][:cutoff_index+1].tolist(),
                'sampled_past_counts':sample_count[group][:cutoff_index+1].tolist(),
                'heldout_target_cells':n_full-n_sample,
                'scores':{name:signed_score(core,pred,truth,base,mapped) for name,pred in controls.items()}})
    conclusion={'full_better_both_whole':all(
        r['scores']['full_past_trend']['chance_adjusted_score']>
        r['scores']['sampled_past_trend']['chance_adjusted_score']
        for r in results if r['group']=='whole' and all(s['valid'] for s in r['scores'].values())),
        'cardiac_results_valid':all(all(s['valid'] for s in r['scores'].values()) for r in results if r['group']=='cardiac')}
    public={'created_utc':now(),'scope':'Historical source-only pseudobulk signed-rank screening. Not challenge full-panel score or independent embryos.',
            'source_report_sha256':digest(source/'report.json'),'results':results,'screen':conclusion,
            'limitations':'Target stage source cells excluded from learner stages and sampled target cells removed from held-out truth; labels may not match challenge ontology. Do not fit from this report.'}
    (HERE/'FULL_SOURCE_RANK_TRANSFER_RESULTS.json').write_text(json.dumps(public,indent=2))


if __name__=='__main__':main()
