"""Past-only fixed tanh time bank; mapped-gene development ranking proxy."""
import json
from collections import Counter

import numpy as np
import pandas as pd
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event
from eb_gene_slope_screen import score, K

FOLDS=[(8.25,8.5),(8.5,8.75)]
RATES=[1.,2.,4.]
CENTERS=[-.375,0.]
ALPHAS=[.25,.5]


def contrast(times,cutoff,target,rate=None,center=0.):
    def transform(t):
        return np.asarray(t)-cutoff if rate is None else np.tanh(rate*(np.asarray(t)-cutoff-center))
    basis=np.column_stack([np.ones(len(times)),transform(times)])
    # Difference of fitted endpoints, anchored at the actually observed current mean.
    delta=np.array([0.,float(transform(target)-transform(cutoff))])
    return delta@np.linalg.pinv(basis)


def main():
    out=HERE/'private/saturating_time_proxy_01'
    if out.exists():raise ValueError('Retain frozen prior run')
    source=HERE/'private/associated_prepared_01'
    source_report=json.loads((source/'report.json').read_text())
    for filename,key in [('expression.npy','expression_sha256'),('genes.csv','genes_sha256'),('selected_metadata.csv','metadata_sha256')]:
        if digest(source/filename)!=source_report[key]:raise ValueError('Prepared source changed')
    panel_path=HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt'
    panel=panel_path.read_text().splitlines();symbols=pd.read_csv(source/'genes.csv').symbol.fillna('').tolist()
    frequency=Counter(symbols);lookup={s:i for i,s in enumerate(symbols) if s and frequency[s]==1}
    mapped=np.array([i for i,s in enumerate(panel) if s in lookup]);columns=np.array([lookup[panel[i]] for i in mapped])
    plan={'created_utc':now(),'code_sha256':digest(HERE/'saturating_time_proxy.py'),
        'score_dependency_sha256':digest(HERE/'eb_gene_slope_screen.py'),
        'source_report_sha256':digest(source/'report.json'),'panel_sha256':digest(panel_path),
        'folds':FOLDS,'past_snapshots_per_fold':4,'rates_per_day':RATES,
        'center_offsets_from_cutoff_days':CENTERS,'shrinkage_alphas':ALPHAS,'seed':None,
        'formula':'OLS intercept+coefficient*tanh(rate*(time-cutoff-center)), fit to four equally weighted source stage mean log1p expressions. Delta=fitted(target)-fitted(cutoff), anchored at observed cutoff. Fixed centers -.375 and0 days; no per-gene parameter selection.',
        'controls':'Copy, recent quarter-day mean delta, matched four-stage linear delta and its .25/.5 blends with recent delta.',
        'gate':'Candidate beats recent and matched-linear control chance-adjusted signed top200 overlap strictly on both folds; partial rank direction >= both controls and >0 on both. All candidates saved, no target-driven grid extensions.',
        'scope':'Mapped-gene rank/direction development proxy only; repeated source folds are not independent embryo validation. No external data or full-panel skill/reward.',
        'resources':'D-only source memmap256genechunks, small contrasts/prediction vectors/split rows retained.'}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (out/'executed_source.py').write_bytes((HERE/'saturating_time_proxy.py').read_bytes())
    events=out/'events.jsonl';append_event(events,'plan_frozen',plan_sha256=digest(out/'plan.json'))
    # Verify the linear contrast recovers a known linear signal and removes constants.
    times=np.array([-.75,-.5,-.25,0.]);test=contrast(times,0.,.25)
    if not np.allclose([test@np.ones(4),test@(3+2*times)],[0.,.5],atol=1e-12):raise ValueError('Linear contrast validation failed')
    x=np.load(source/'expression.npy',mmap_mode='r');meta=pd.read_csv(source/'selected_metadata.csv');stages=meta.numeric_stage.to_numpy(float)
    outcomes=[]
    for cutoff,target in FOLDS:
        times=cutoff+np.array([-.75,-.5,-.25,0.])
        groups=[np.flatnonzero(stages==t) for t in times]
        if any(len(g)<100 for g in groups):raise ValueError('Insufficient four-stage support')
        np.savez_compressed(out/f'past_rows_{cutoff}.npz',**{str(t):g for t,g in zip(times,groups)})
        curves={f'tanh_rate{rate}_center{center}':contrast(times,cutoff,target,rate,center) for rate in RATES for center in CENTERS}
        linear=contrast(times,cutoff,target)
        for name,c in curves.items():
            if not np.isfinite(c).all() or abs(c.sum())>1e-10:raise ValueError('Curve contrast invalid')
        means=np.empty((4,len(mapped)))
        for start in range(0,len(mapped),256):
            sl=slice(start,min(start+256,len(mapped)))
            for i,rows in enumerate(groups):means[i,sl]=np.asarray(x[np.ix_(rows,columns[sl])],dtype=np.float32).mean(0,dtype=np.float64)
        ref=means[-1];recent=ref-means[-2];ols=linear@means
        candidates={'copy':np.zeros_like(ref),'recent_linear':recent,'past_four_stage_linear':ols}
        controls={}
        for alpha in ALPHAS:candidates[f'linear_blend_{alpha}']=(1-alpha)*recent+alpha*ols
        for name,c in curves.items():
            curve=c@means;candidates[name]=curve;controls[name]='past_four_stage_linear'
            for alpha in ALPHAS:
                blend=f'{name}_blend{alpha}';candidates[blend]=(1-alpha)*recent+alpha*curve;controls[blend]=f'linear_blend_{alpha}'
        frozen=out/f'deltas_{cutoff}.npz'
        np.savez_compressed(frozen,mapped=mapped,linear_contrast=linear,**candidates)
        append_event(events,'predictions_frozen_before_target',cutoff=cutoff,target=target,sha256=digest(frozen))
        target_rows=np.flatnonzero(stages==target);truth=np.empty(len(mapped))
        for start in range(0,len(mapped),256):
            sl=slice(start,min(start+256,len(mapped)))
            truth[sl]=np.asarray(x[np.ix_(target_rows,columns[sl])],dtype=np.float32).mean(0,dtype=np.float64)-ref[sl]
        order=np.argsort(truth,kind='stable');up,down=order[-K:],order[:K]
        results={name:score(delta,truth,ref,up,down) for name,delta in candidates.items()}
        row={'cutoff':cutoff,'target':target,'past_stages':times.tolist(),'past_stage_counts':[len(g) for g in groups],
             'target_rows':len(target_rows),'curve_contrasts':{k:v.tolist() for k,v in curves.items()},'results':results}
        outcomes.append(row);append_event(events,'proxy_fold_scored',cutoff=cutoff,results=results)
    passing=[]
    for name,control in controls.items():
        if all(f['results'][name]['chance_adjusted_overlap']>max(f['results'][control]['chance_adjusted_overlap'],f['results']['recent_linear']['chance_adjusted_overlap'])
            and f['results'][name]['partial_spearman']>=max(f['results'][control]['partial_spearman'],f['results']['recent_linear']['partial_spearman'])
            and f['results'][name]['partial_spearman']>0 for f in outcomes):passing.append(name)
    report={'updated_utc':now(),'plan_sha256':digest(out/'plan.json'),'outcomes':outcomes,'passing_candidates':passing,
        'proxy_evaluations':sum(len(f['results']) for f in outcomes),'mapped_panel_genes':len(mapped),
        'full_panel_scores':0,'reward_delta':0,'independent_embryo_validation':False,
        'next_action':'Freeze matched full-panel ablation for passing curve' if passing else 'Reject this fixed saturating bank; next declared source-only state-by-time interaction with strong shrinkage and matched global controls. Scientific nonlinear family remains open.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    public=dict(report,report_sha256=digest(out/'report.json'))
    (HERE/'SATURATING_TIME_PROXY_RESULTS.json').write_text(json.dumps(public,indent=2)+'\n')
    append_event(events,'screen_completed',report_sha256=public['report_sha256'],passing_candidates=passing)
    print(json.dumps({k:v for k,v in public.items() if k!='outcomes'}))


if __name__=='__main__':main()
