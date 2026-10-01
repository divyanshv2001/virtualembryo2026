"""Partially pooled categorical lineage-time ridge slopes; source-only proxy."""
import json
from collections import Counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event
from lineage_residual_screen import lineage
from eb_gene_slope_screen import score, K

FOLDS=[(8.,8.25),(8.25,8.5)]
PENALTIES=[100.,1000.]
ALPHAS=[.25,.5]
MIN_CELLS=20


def contrast(design,delta,penalties):
    return delta@np.linalg.solve(design.T@design+np.diag(penalties),design.T)


def main():
    out=HERE/'private/lineage_time_partial_pool_proxy_01'
    if out.exists():raise ValueError('Retain prior frozen run')
    source=HERE/'private/associated_prepared_01';manifest=json.loads((source/'report.json').read_text())
    for name,key in [('expression.npy','expression_sha256'),('genes.csv','genes_sha256'),('selected_metadata.csv','metadata_sha256')]:
        if digest(source/name)!=manifest[key]:raise ValueError('Atlas source changed')
    panel_path=HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt'
    panel=panel_path.read_text().splitlines();symbols=pd.read_csv(source/'genes.csv').symbol.fillna('').tolist()
    freq=Counter(symbols);lookup={s:i for i,s in enumerate(symbols) if s and freq[s]==1}
    mapped=np.array([i for i,s in enumerate(panel) if s in lookup]);columns=np.array([lookup[panel[i]] for i in mapped])
    plan={'created_utc':now(),'code_sha256':digest(HERE/'lineage_time_partial_pool_proxy.py'),
        'dependencies_sha256':{n:digest(HERE/n) for n in ('lineage_residual_screen.py','eb_gene_slope_screen.py')},
        'source_report_sha256':digest(source/'report.json'),'panel_sha256':digest(panel_path),
        'folds':FOLDS,'past_snapshots':3,'minimum_lineage_cells_each_past_stage':MIN_CELLS,
        'deviation_penalties':PENALTIES,'recent_blend_alphas':ALPHAS,'seed':None,
        'formula':'Log1p expression ridge on full lineage one-hot intercepts (unpenalized), global centeredtime slope (penalty1) and lineage*time deviations (100/1000). Quarterday delta at fixed current lineage fractions: global slope +currentfraction weighted lineage deviations. Unsupported groups collapse to fallback using paststage counts only.',
        'controls':'Copy/recentlinear, pooled three-stage OLS, lineage-intercept globaltime model without deviations and matched recentblends.',
        'gate':'Bothfolds strict chance-adjusted signedtop200 overlap gain versusrecent and matched global/OLS controls; partial rank direction>=all controls and >0. No target-driven penalty choices.',
        'scope':'Source-only mapped-gene development screen, no fullpanel score/reward. Supplied lineage labels do not establish independent embryo replication or causal effects.',
        'resources':'D-only memmap256genechunks,twoBLASthreads,small frozen splits/contrasts/predictions.'}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (out/'executed_source.py').write_bytes((HERE/'lineage_time_partial_pool_proxy.py').read_bytes())
    events=out/'events.jsonl';append_event(events,'plan_frozen',plan_sha256=digest(out/'plan.json'))
    x=np.load(source/'expression.npy',mmap_mode='r');meta=pd.read_csv(source/'selected_metadata.csv')
    stages=meta.numeric_stage.to_numpy(float);labels=meta.celltype_extended_atlas.map(lineage).to_numpy()
    outcomes=[]
    for cutoff,target in FOLDS:
        times=[cutoff-.5,cutoff-.25,cutoff];fit=np.flatnonzero(np.isin(stages,times))
        stage=stages[fit];current=stage==cutoff;previous=stage==cutoff-.25
        support={g:[int(((labels[fit]==g)&(stage==t)).sum()) for t in times] for g in sorted(set(labels[fit]))}
        viable={g for g,counts in support.items() if min(counts)>=MIN_CELLS}
        group_labels=np.array([g if g in viable else 'fallback' for g in labels[fit]])
        groups=sorted(set(group_labels));onehot=np.column_stack([(group_labels==g).astype(float) for g in groups])
        if np.any(onehot.sum(0)==0):raise ValueError('Empty lineage design')
        t=(stage-cutoff)/.25;t-=t.mean();fractions=onehot[current].mean(0)
        global_design=np.column_stack([onehot,t]);global_eval=np.r_[np.zeros(len(groups)),1.]
        global_penalty=np.r_[np.zeros(len(groups)),1.]
        contrasts={'global':contrast(global_design,global_eval,global_penalty)}
        design=np.column_stack([global_design,onehot*t[:,None]])
        evaluation=np.r_[global_eval,fractions]
        for penalty in PENALTIES:
            contrasts[f'lineage_{penalty}']=contrast(design,evaluation,np.r_[global_penalty,np.full(len(groups),penalty)])
        contrasts['past_ols']=contrast(np.column_stack([np.ones(len(fit)),t]),np.array([0.,1.]),[0.,0.])
        for name,c in contrasts.items():
            if not np.isfinite(c).all() or abs(c.sum())>1e-9:raise ValueError('Invalid constant-response contrast')
            if name!='past_ols' and not np.allclose(c@onehot,0.,atol=1e-9):raise ValueError('Static lineage intercept leaks into temporal delta')
        fixed=out/f'contrasts_{cutoff}.npz';np.savez_compressed(fixed,fit_rows=fit,groups=np.array(groups),current_fractions=fractions,**contrasts)
        ref=np.empty(len(mapped));recent=np.empty(len(mapped));deltas={k:np.empty(len(mapped)) for k in contrasts}
        for start in range(0,len(mapped),256):
            sl=slice(start,min(start+256,len(mapped)));values=np.asarray(x[np.ix_(fit,columns[sl])],dtype=np.float32)
            ref[sl]=values[current].mean(0,dtype=np.float64);recent[sl]=ref[sl]-values[previous].mean(0,dtype=np.float64)
            for name,c in contrasts.items():deltas[name][sl]=c@values
        candidates={'copy':np.zeros_like(ref),'recent_linear':recent,**deltas}
        for alpha in ALPHAS:
            for name,delta in deltas.items():candidates[f'{name}_blend{alpha}']=(1-alpha)*recent+alpha*delta
        controls={}
        for penalty in PENALTIES:
            controls[f'lineage_{penalty}']=['global','past_ols']
            for alpha in ALPHAS:controls[f'lineage_{penalty}_blend{alpha}']=[f'global_blend{alpha}',f'past_ols_blend{alpha}']
        frozen=out/f'deltas_{cutoff}.npz';np.savez_compressed(frozen,mapped=mapped,**candidates)
        append_event(events,'predictions_frozen_before_target',cutoff=cutoff,target=target,contrasts_sha256=digest(fixed),predictions_sha256=digest(frozen))
        rows=np.flatnonzero(stages==target);truth=np.empty(len(mapped))
        for start in range(0,len(mapped),256):
            sl=slice(start,min(start+256,len(mapped)))
            truth[sl]=np.asarray(x[np.ix_(rows,columns[sl])],dtype=np.float32).mean(0,dtype=np.float64)-ref[sl]
        order=np.argsort(truth,kind='stable');up,down=order[-K:],order[:K]
        scores={name:score(delta,truth,ref,up,down) for name,delta in candidates.items()}
        row={'cutoff':cutoff,'target':target,'lineage_counts_by_past_stage':support,'retained_groups':groups,
             'current_fractions':dict(zip(groups,fractions.tolist())),'results':scores}
        outcomes.append(row);append_event(events,'proxy_fold_scored',cutoff=cutoff,results=scores)
    passing=[]
    for name,matched in controls.items():
        names=['recent_linear',*matched]
        if all(f['results'][name]['chance_adjusted_overlap']>max(f['results'][c]['chance_adjusted_overlap'] for c in names)
            and f['results'][name]['partial_spearman']>=max(f['results'][c]['partial_spearman'] for c in names)
            and f['results'][name]['partial_spearman']>0 for f in outcomes):passing.append(name)
    report={'updated_utc':now(),'plan_sha256':digest(out/'plan.json'),'outcomes':outcomes,'passing_candidates':passing,
        'proxy_evaluations':sum(len(f['results']) for f in outcomes),'invariants_passed':['finite_zero_sum','static_lineage_intercept_invariance'],
        'full_panel_scores':0,'reward_delta':0,'independent_embryo_validation':False,
        'next_action':'Freeze matched full-panel ablation for passing lineage slope' if passing else 'Reject tested lineage-time partial pooling. Audit forecast horizon and scorer-aligned proxy fidelity before more quarter-day slope variants; scientific families remain open.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    public=dict(report,report_sha256=digest(out/'report.json'))
    (HERE/'LINEAGE_TIME_PARTIAL_POOL_RESULTS.json').write_text(json.dumps(public,indent=2)+'\n')
    append_event(events,'screen_completed',report_sha256=public['report_sha256'],passing_candidates=passing)
    print(json.dumps({k:v for k,v in public.items() if k!='outcomes'}))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
