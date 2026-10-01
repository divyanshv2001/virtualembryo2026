"""Conditional time coefficient via ridge Schur complement, past-only proxy.

This is a mapped-gene ranking screen, not a T1 full-panel predictive score.
"""
import json
from collections import Counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event
from eb_gene_slope_screen import score, K
from lineage_residual_screen import lineage

FOLDS=[(8.,8.25),(8.25,8.5)]
ALPHAS=[.1,.25]


def temporal_contrast(controls, time, ridge=1.):
    """Exact time-coefficient contrast for penalized joint regression [W,t]."""
    penalty=np.eye(controls.shape[1])*ridge
    penalty[0,0]=0
    projection=np.linalg.solve(controls.T@controls+penalty,controls.T@time)
    remainder=time-controls@projection
    denominator=float(time@time+ridge-time@controls@projection)
    if denominator<=1e-8:raise ValueError('Unidentifiable conditional time coefficient')
    return remainder/denominator,{'schur_denominator':denominator,
        'unpenalized_time_energy':float(time@time),
        'residual_time_energy':float(remainder@remainder),
        'residual_time_energy_fraction':float((remainder@remainder)/max(time@time,1e-12))}


def main():
    source=HERE/'private/associated_prepared_01'
    archive=HERE/'private/cnf_hurdle_temporal_01'
    out=HERE/'private/latent_residual_gene_screen_01'
    if out.exists():raise ValueError('Preserve frozen original run')
    manifest=json.loads((source/'report.json').read_text())
    for filename,key in [('expression.npy','expression_sha256'),
                         ('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(source/filename)!=manifest[key]:raise ValueError('Prepared source changed')
    panel_path=HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt'
    panel=panel_path.read_text().splitlines()
    symbols=pd.read_csv(source/'genes.csv').symbol.fillna('').tolist()
    counts=Counter(symbols);lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
    mapped=np.array([i for i,s in enumerate(panel) if s in lookup])
    atlas=np.array([lookup[panel[i]] for i in mapped])
    plan={'created_utc':now(),'folds':FOLDS,'alphas':ALPHAS,'ridge':1.,
        'code_sha256':digest(HERE/'latent_residual_gene_screen.py'),
        'dependency_sha256':{name:digest(HERE/name) for name in ['eb_gene_slope_screen.py','lineage_residual_screen.py']},
        'prepared_report_sha256':digest(source/'report.json'),'panel_sha256':digest(panel_path),
        'encoder_sha256':{str(c):digest(archive/f'cutoff_{c}'/'encoder.npz') for c,_ in FOLDS},
        'fit':'Three source snapshots at cutoff-.5,-.25,0. Frozen encoder trained through each cutoff; standardized8D state and supplied lineage one-hot controls, unpenalized intercept, ridge1 controls/time. Schur-complement conditional time coefficient; units per .25day. Log1p full-expression response includes zeros.',
        'controls':'Persistence, recent pooled slope, past three-stage pooled OLS and matched .1/.25 OLS blends. Candidate blends recent pooled delta with conditional time delta at .1/.25.',
        'gate':'Strict signed top200 chance-adjusted overlap gain on both folds versus recent-linear AND matched OLS blend; partial Spearman no worse than either and >persistence on both. Reject if residual time energy fraction<.05 on either fold.',
        'scope':'Mapped-gene historical rank/direction proxy only; no full-panel raw metrics or skill/headline, no reward. Atlas annotation controls may reflect joint annotation; conditional slopes are associations, not causal velocities or independent embryo validation.',
        'retention':'Stream256genes; retain small contrasts/splits/deltas/events, no full forecast matrices.',
        'official_submissions':0,'jev_requests':0}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (out/'executed_source.py').write_bytes((HERE/'latent_residual_gene_screen.py').read_bytes())
    events=out/'events.jsonl';append_event(events,'plan_frozen',sha256=digest(out/'plan.json'))
    x=np.load(source/'expression.npy',mmap_mode='r')
    meta=pd.read_csv(source/'selected_metadata.csv');stages=meta.numeric_stage.to_numpy(float)
    labels=meta.celltype_extended_atlas.map(lineage).to_numpy()
    outcomes=[]
    for cutoff,target in FOLDS:
        fit=np.flatnonzero(np.isin(stages,[cutoff-.5,cutoff-.25,cutoff]))
        np.save(out/f'fit_rows_{cutoff}.npy',fit)
        encoder_path=archive/f'cutoff_{cutoff}'/'encoder.npz'
        assert digest(encoder_path)==plan['encoder_sha256'][str(cutoff)]
        with np.load(encoder_path) as e:
            feat=np.array([lookup[panel[i]] for i in e['features']])
            raw=np.asarray(x[np.ix_(fit,feat)],dtype=np.float32)
            z=((raw-e['center'])/e['scale']-e['pca_center'])@e['basis'].T
        del raw
        z=(z-z.mean(0))/np.maximum(z.std(0),1e-6)
        groups=sorted(set(labels[fit]))
        onehot=np.column_stack([(labels[fit]==g).astype(float) for g in groups[1:]])
        controls=np.column_stack([np.ones(len(fit)),z,onehot])
        t=(stages[fit]-cutoff)/.25;t-=t.mean()
        contrast,diagnostic=temporal_contrast(controls,t)
        pooled,_=temporal_contrast(np.ones((len(fit),1)),t)
        np.savez_compressed(out/f'contrast_{cutoff}.npz',fit_rows=fit,conditional=contrast,pooled=pooled,groups=np.array(groups))
        ref=np.empty(len(atlas));recent=np.empty(len(atlas));b=np.empty(len(atlas));ols=np.empty(len(atlas))
        current=stages[fit]==cutoff;previous=stages[fit]==cutoff-.25
        for start in range(0,len(atlas),256):
            sl=slice(start,min(start+256,len(atlas)))
            values=np.asarray(x[np.ix_(fit,atlas[sl])],dtype=np.float32)
            ref[sl]=values[current].mean(0,dtype=np.float64)
            recent[sl]=ref[sl]-values[previous].mean(0,dtype=np.float64)
            b[sl]=contrast@values;ols[sl]=pooled@values
        candidates={'copy':np.zeros_like(ref),'recent_linear':recent,'past_ols':ols}
        for alpha in ALPHAS:
            candidates[f'residual_{alpha}']=(1-alpha)*recent+alpha*b
            candidates[f'ols_blend_{alpha}']=(1-alpha)*recent+alpha*ols
        frozen=out/f'deltas_{cutoff}.npz'
        np.savez_compressed(frozen,mapped=mapped,**{k:v.astype(np.float32) for k,v in candidates.items()})
        append_event(events,'predictions_frozen_before_target_read',cutoff=cutoff,target=target,
                     deltas_sha256=digest(frozen),contrast_sha256=digest(out/f'contrast_{cutoff}.npz'),time_diagnostic=diagnostic)
        rows=np.flatnonzero(stages==target);truth=np.empty(len(atlas))
        for start in range(0,len(atlas),256):
            sl=slice(start,min(start+256,len(atlas)))
            truth[sl]=np.asarray(x[np.ix_(rows,atlas[sl])],dtype=np.float32).mean(0,dtype=np.float64)-ref[sl]
        order=np.argsort(truth,kind='stable');up,down=order[-K:],order[:K]
        results={name:score(delta,truth,ref,up,down) for name,delta in candidates.items()}
        outcomes.append({'cutoff':cutoff,'target':target,'fit_rows':len(fit),'time_diagnostic':diagnostic,'results':results})
        append_event(events,'proxy_fold_scored',cutoff=cutoff,results=results)
    passing=[]
    for alpha in ALPHAS:
        name=f'residual_{alpha}';other=f'ols_blend_{alpha}'
        if all(f['time_diagnostic']['residual_time_energy_fraction']>=.05
            and f['results'][name]['chance_adjusted_overlap']>max(f['results']['recent_linear']['chance_adjusted_overlap'],f['results'][other]['chance_adjusted_overlap'])
            and f['results'][name]['partial_spearman']>=max(f['results']['recent_linear']['partial_spearman'],f['results'][other]['partial_spearman'])
            and f['results'][name]['partial_spearman']>0 for f in outcomes):passing.append(name)
    report={'status':'completed','outcomes':outcomes,'passing_candidates':passing,
            'plan_sha256':digest(out/'plan.json'),'full_panel_scores':0,'reward_delta':0,
            'scope':plan['scope'],'official_score':None}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    public=dict(report,updated_utc=now(),report_sha256=digest(out/'report.json'))
    (HERE/'LATENT_RESIDUAL_GENE_SCREEN_RESULTS.json').write_text(json.dumps(public,indent=2)+'\n')
    append_event(events,'screen_completed',passing_candidates=passing,report_sha256=public['report_sha256'])
    print(json.dumps(public))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
