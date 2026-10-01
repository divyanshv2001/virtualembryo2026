"""Past-frozen latent-state by time ridge interactions; development proxy."""
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
INTERACTION_PENALTIES=[10000.,100000.]
ALPHAS=[.1,.25]


def fitted_contrast(design, evaluation_delta, penalties):
    return evaluation_delta@np.linalg.solve(design.T@design+np.diag(penalties),design.T)


def main():
    out=HERE/'private/state_time_interaction_proxy_01'
    if out.exists():raise ValueError('Keep prior original screen')
    source=HERE/'private/associated_prepared_01'
    archive=HERE/'private/cnf_hurdle_temporal_01'
    manifest=json.loads((source/'report.json').read_text())
    for filename,key in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(source/filename)!=manifest[key]:raise ValueError('Prepared source changed')
    panel_path=HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt'
    panel=panel_path.read_text().splitlines();symbols=pd.read_csv(source/'genes.csv').symbol.fillna('').tolist()
    frequency=Counter(symbols);lookup={s:i for i,s in enumerate(symbols) if s and frequency[s]==1}
    mapped=np.array([i for i,s in enumerate(panel) if s in lookup]);columns=np.array([lookup[panel[i]] for i in mapped])
    plan={'created_utc':now(),'code_sha256':digest(HERE/'state_time_interaction_proxy.py'),
        'dependencies_sha256':{n:digest(HERE/n) for n in ('lineage_residual_screen.py','eb_gene_slope_screen.py')},
        'prepared_report_sha256':digest(source/'report.json'),'panel_sha256':digest(panel_path),
        'encoder_sha256':{str(c):digest(archive/f'cutoff_{c}'/'encoder.npz') for c,_ in FOLDS},
        'folds':FOLDS,'past_stages':3,'alphas':ALPHAS,'interaction_penalties':INTERACTION_PENALTIES,'seed':None,
        'formula':'Ridge log1p expression on [intercept, standardized frozen8Dstate, lineage onehot, centeredtime, state*time]. Intercept unpenalized, static/time penalty1, interaction10000/100000. Contrast predicts quarterday change at fixed observed current-state mean: delta=[0static,1time,mean_current_state interactions].',
        'controls':'Copy/recentlinear, pooledthree-stageOLS, matchedglobalconditionaltime ridge without interactions and their .1/.25recentblends.',
        'gate':'Bothfolds strict chance-adjusted signedtop200 overlap gain versusrecent and matchedglobal/pooledlinear control; direction >= all three and >0. No target-driven fit choices.',
        'scope':'Source-only mapped-gene development proxy; supplied labels/pooledcellcounts are not independent embryo validation, interactions not causal velocities. No external data/fullpanel score/reward.',
        'resources':'D-only memmap256gene responsechunks;twoBLASthreads, retain small contrast/prediction/split artifacts.'}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (out/'executed_source.py').write_bytes((HERE/'state_time_interaction_proxy.py').read_bytes())
    events=out/'events.jsonl';append_event(events,'plan_frozen',plan_sha256=digest(out/'plan.json'))
    # Constant-response invariance and zero interactions exactly reducing to global model.
    t=np.linspace(-1,1,101);d=np.column_stack([np.ones(101),t]);c=fitted_contrast(d,np.array([0.,1.]),[0.,1.])
    extra=np.column_stack([d,np.zeros(101)]);c0=fitted_contrast(extra,np.array([0.,1.,0.]),[0.,1.,10000.])
    if abs(c.sum())>1e-12 or not np.allclose(c,c0,atol=1e-12):raise ValueError('Contrast invariance checks failed')
    x=np.load(source/'expression.npy',mmap_mode='r');meta=pd.read_csv(source/'selected_metadata.csv')
    stages=meta.numeric_stage.to_numpy(float);labels=meta.celltype_extended_atlas.map(lineage).to_numpy()
    outcomes=[]
    for cutoff,target in FOLDS:
        fit=np.flatnonzero(np.isin(stages,[cutoff-.5,cutoff-.25,cutoff]));current=stages[fit]==cutoff;previous=stages[fit]==cutoff-.25
        with np.load(archive/f'cutoff_{cutoff}'/'encoder.npz') as e:
            features=np.array([lookup[panel[i]] for i in e['features']])
            raw=np.asarray(x[np.ix_(fit,features)],dtype=np.float32)
            z=((raw-e['center'])/e['scale']-e['pca_center'])@e['basis'].T
        del raw
        z=(z-z.mean(0))/np.maximum(z.std(0),1e-6)
        groups=sorted(set(labels[fit]));onehot=np.column_stack([(labels[fit]==g).astype(float) for g in groups[1:]])
        static=np.column_stack([np.ones(len(fit)),z,onehot]);t=(stages[fit]-cutoff)/.25;t-=t.mean()
        global_design=np.column_stack([static,t]);global_penalty=np.ones(global_design.shape[1]);global_penalty[0]=0
        global_eval=np.zeros(global_design.shape[1]);global_eval[-1]=1.
        global_contrast=fitted_contrast(global_design,global_eval,global_penalty)
        design=np.column_stack([global_design,z*t[:,None]])
        evaluation=np.r_[global_eval,z[current].mean(0)]
        contrasts={'global':global_contrast}
        for penalty in INTERACTION_PENALTIES:
            contrasts[f'interaction_{penalty}']=fitted_contrast(design,evaluation,np.r_[global_penalty,np.full(z.shape[1],penalty)])
        pooled_design=np.column_stack([np.ones(len(fit)),t])
        contrasts['past_ols']=fitted_contrast(pooled_design,np.array([0.,1.]),[0.,0.])
        if any(not np.isfinite(v).all() or abs(v.sum())>1e-9 for v in contrasts.values()):raise ValueError('Nonfinite/constant-variant contrast')
        contrast_path=out/f'contrasts_{cutoff}.npz';np.savez_compressed(contrast_path,fit_rows=fit,current_state_mean=z[current].mean(0),**contrasts)
        ref=np.empty(len(mapped));recent=np.empty(len(mapped));deltas={k:np.empty(len(mapped)) for k in contrasts}
        for start in range(0,len(mapped),256):
            sl=slice(start,min(start+256,len(mapped)));values=np.asarray(x[np.ix_(fit,columns[sl])],dtype=np.float32)
            ref[sl]=values[current].mean(0,dtype=np.float64);recent[sl]=ref[sl]-values[previous].mean(0,dtype=np.float64)
            for k,c in contrasts.items():deltas[k][sl]=c@values
        candidates={'copy':np.zeros_like(ref),'recent_linear':recent,**deltas};matched_controls={}
        for alpha in ALPHAS:
            for name,delta in deltas.items():candidates[f'{name}_blend{alpha}']=(1-alpha)*recent+alpha*delta
        for penalty in INTERACTION_PENALTIES:
            name=f'interaction_{penalty}';matched_controls[name]=['global','past_ols']
            for alpha in ALPHAS:matched_controls[f'{name}_blend{alpha}']=[f'global_blend{alpha}',f'past_ols_blend{alpha}']
        frozen=out/f'deltas_{cutoff}.npz';np.savez_compressed(frozen,mapped=mapped,**candidates)
        append_event(events,'predictions_frozen_before_target',cutoff=cutoff,target=target,contrast_sha256=digest(contrast_path),predictions_sha256=digest(frozen))
        rows=np.flatnonzero(stages==target);truth=np.empty(len(mapped))
        for start in range(0,len(mapped),256):
            sl=slice(start,min(start+256,len(mapped)))
            truth[sl]=np.asarray(x[np.ix_(rows,columns[sl])],dtype=np.float32).mean(0,dtype=np.float64)-ref[sl]
        order=np.argsort(truth,kind='stable');up,down=order[-K:],order[:K]
        results={name:score(pred,truth,ref,up,down) for name,pred in candidates.items()}
        row={'cutoff':cutoff,'target':target,'fit_rows':len(fit),'current_state_mean':z[current].mean(0).tolist(),'results':results}
        outcomes.append(row);append_event(events,'proxy_fold_scored',cutoff=cutoff,results=results)
    passing=[]
    for name,controls in matched_controls.items():
        control_names=['recent_linear',*controls]
        if all(f['results'][name]['chance_adjusted_overlap']>max(f['results'][c]['chance_adjusted_overlap'] for c in control_names)
            and f['results'][name]['partial_spearman']>=max(f['results'][c]['partial_spearman'] for c in control_names)
            and f['results'][name]['partial_spearman']>0 for f in outcomes):passing.append(name)
    report={'updated_utc':now(),'plan_sha256':digest(out/'plan.json'),'outcomes':outcomes,'passing_candidates':passing,
        'proxy_evaluations':sum(len(f['results']) for f in outcomes),'full_panel_scores':0,'reward_delta':0,
        'invariant_checks_passed':['constant_response','zero_interaction_reduces_to_global','finite_zero_sum_contrasts'],
        'independent_embryo_validation':False,
        'next_action':'Freeze matched full-panel ablation for passing interaction' if passing else 'Reject this latent state*time bank; next declared lineage-specific temporal interaction with partial pooling or transport uncertainty calibration. Entire state-dependent scientific family remains open.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    public=dict(report,report_sha256=digest(out/'report.json'))
    (HERE/'STATE_TIME_INTERACTION_PROXY_RESULTS.json').write_text(json.dumps(public,indent=2)+'\n')
    append_event(events,'screen_completed',report_sha256=public['report_sha256'],passing_candidates=passing)
    print(json.dumps({k:v for k,v in public.items() if k!='outcomes'}))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
