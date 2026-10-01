"""Current-observation-only feasibility, not a temporal forecast or score."""
import json
from collections import Counter
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now,append_event
from lineage_residual_screen import lineage
from neural_hurdle_forecast import systematic_bernoulli
from support_mass_bounds import necessary_bounds

RUN='support_mass_bounds_01'


def main():
    source=HERE/'private/associated_prepared_01'; out=HERE/'private'/RUN
    if out.exists(): raise ValueError('Preserve original audit')
    prepared=json.loads((source/'report.json').read_text())
    for f,k in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(source/f)!=prepared[k]: raise ValueError('Prepared source changed: '+f)
    panel_path=HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt'
    panel=panel_path.read_text().splitlines(); symbols=pd.read_csv(source/'genes.csv').symbol.fillna('').tolist()
    counts=Counter(symbols); lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
    mapped=np.array([i for i,s in enumerate(panel) if s in lookup]);atlas=np.array([lookup[panel[i]] for i in mapped])
    metadata=pd.read_csv(source/'selected_metadata.csv'); stages=metadata.numeric_stage.to_numpy(float)
    samples=metadata['sample'].astype(str).to_numpy();labels=metadata.celltype_extended_atlas.map(lineage).to_numpy()
    plan={'created_utc':now(),'code_sha256':digest(HERE/'support_mass_bounds_audit.py'),
          'bounds_sha256':digest(HERE/'support_mass_bounds.py'),'prepared_report_sha256':digest(source/'report.json'),
          'panel_sha256':digest(panel_path),'route_sha256':digest(HERE/'JEV_PROJECTION_FAILURE_ROUTE_20261001_01.json'),
          'observations':[[8.,'16'],[8.,'33'],[8.25,'25'],[8.25,'24']],
          'seed':20260928,'alpha':.25,'bound_tolerance':1e-8,
          'support':'CM current expression only, mapped genes with >=2 positives in heldcapture; unsupported genes unchanged. Lock post-switch support; explicit infeasibility, never fallback counted as success.',
          'method':'Current-only synthetic odds perturbation: Jeffreys current-CM empirical q plus .25*(logit held-q minus logit othercapture-q). Systematic switches; added values use current-held conditional-positive log means. Compute necessary raw-mass Jensen lower bounds and per-column supported-row log capacity for original current margins. Passing does NOT prove feasibility.',
          'scope':'Numerical feasibility of joint constraints on currentobservations only, not CNFforecast,temporalvalidation,DErankguarantee,officialscore or reward. Entirely stages<=8.25. No future expression or future-score-driven fitting. No independent embryo identity.',
          'decision':'A failed necessary bound proves incompatibility of the locked support/margins; all passing cases remain unresolved. No temporal scoring is authorized by a passing bound alone.'}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2));events=out/'events.jsonl'
    append_event(events,'plan_frozen',sha256=digest(out/'plan.json'))
    x=np.load(source/'expression.npy',mmap_mode='r');results=[]
    for cutoff,held in plan['observations']:
        rows=np.flatnonzero((stages==cutoff)&(samples==held)&(labels=='cardiomyocyte'))
        other=np.flatnonzero((stages==cutoff)&(samples!=held)&(labels=='cardiomyocyte'))
        if min(len(rows),len(other))<20: raise ValueError('Insufficient current CM support')
        reference=np.asarray(x[np.ix_(rows,atlas)],float);candidate=reference.copy()
        np.save(out/f'rows_{cutoff}_{held}.npy',rows)
        rng=np.random.default_rng(plan['seed']);added_total=removed_total=0
        for start in range(0,len(atlas),512):
            sl=slice(start,min(start+512,len(atlas)));block=reference[:,sl]
            positives=block>0;number=positives.sum(0);supported=number>=2
            q0=(number+.5)/(len(rows)+1.)
            source_q=((x[np.ix_(other,atlas[sl])]>0).sum(0)+.5)/(len(other)+1.)
            q1=expit(logit(q0)+plan['alpha']*(logit(q0)-logit(source_q)))
            change=np.clip(q1-q0,-.25,.25)*supported
            additions=systematic_bernoulli(np.where(~positives,np.maximum(change,0)/(1-q0),0),rng)
            removals=systematic_bernoulli(np.where(positives,np.maximum(-change,0)/q0,0),rng)
            mean_positive=block.sum(0)/np.maximum(number,1)
            updated=block.copy(); updated[additions]=np.broadcast_to(mean_positive,block.shape)[additions];updated[removals]=0
            candidate[:,sl]=updated;added_total+=int(additions.sum());removed_total+=int(removals.sum())
        projected=candidate; audit=necessary_bounds(candidate,reference,tolerance=plan['bound_tolerance']); control=necessary_bounds(reference,reference,tolerance=plan['bound_tolerance'])
        if control['bounds_rejected']: raise ValueError('Known-feasible reference falsely rejected')
        result={'cutoff':cutoff,'held_capture':held,'current_cm_cells':len(rows),'source_cm_cells':len(other),
                'mapped_genes':len(atlas),'added_detections':added_total,'removed_detections':removed_total,
                'finite_nonnegative':bool(np.isfinite(projected).all() and np.all(projected>=0)),
                'necessary_bounds':audit,'reference_control':control,'rows_sha256':digest(out/f'rows_{cutoff}_{held}.npy')}
        results.append(result);append_event(events,'case_completed',**result)
        del reference,candidate,projected
    report={'status':'completed','plan':plan,'results':results,'all_cases_bounds_not_rejected':all(not r['necessary_bounds']['bounds_rejected'] and r['finite_nonnegative'] for r in results),
            'official_score':None,'headline_score':None,'raw_metrics':None,'skills':None,'reward_delta':0}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    public={**report,'plan_sha256':digest(out/'plan.json'),'report_sha256':digest(out/'report.json')}
    (HERE/'SUPPORT_MASS_BOUNDS_RESULTS.json').write_text(json.dumps(public,indent=2))
    append_event(events,'audit_completed',all_cases_bounds_not_rejected=report['all_cases_bounds_not_rejected'])


if __name__=='__main__':
    try:
        with threadpool_limits(limits=2):main()
    except Exception as exc:
        out=HERE/'private'/RUN
        if out.exists():append_event(out/'events.jsonl','run_failed',exception_type=type(exc).__name__,message=str(exc))
        raise
