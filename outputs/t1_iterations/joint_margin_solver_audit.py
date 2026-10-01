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
from joint_margin_solver import solve
from test_joint_margin_solver import main as validate_controls

RUN='joint_margin_solver_01'


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
    plan={'created_utc':now(),'code_sha256':digest(HERE/'joint_margin_solver_audit.py'),
          'solver_sha256':digest(HERE/'joint_margin_solver.py'),'controls_sha256':digest(HERE/'test_joint_margin_solver.py'),'predeclaration_sha256':digest(HERE/'NEXT_JOINT_SOLVER_EXPERIMENT.json'),'prepared_report_sha256':digest(source/'report.json'),
          'panel_sha256':digest(panel_path),'route_sha256':digest(HERE/'JEV_JOINT_SOLVER_ROUTE_20261001_01.json'),
          'observations':[[8.,'16'],[8.,'33'],[8.25,'25'],[8.25,'24']],
          'seed':20260928,'alpha':.25,'iterations':100,'mean_tolerance':1e-5,'raw_mass_relative_tolerance':1e-5,
          'support':'CM current expression only, mapped genes with >=2 positives in heldcapture; unsupported genes unchanged. Lock post-switch support; explicit infeasibility, never fallback counted as success.',
          'method':'Current-only synthetic odds perturbation: Jeffreys current-CM empirical q plus .25*(logit held-q minus logit othercapture-q). Systematic switches; added values use current-held conditional-positive log means. Joint scaled log-gene-mean/raw-row-mass residuals; analytic Schur-cell Gauss-Newton, damping.001, fixed positivity-preserving backoffs1,.5,.25,.125,.0625,.03125 and floor1e-10. No variable-sized dense matrix.',
          'scope':'Numerical feasibility of joint constraints on currentobservations only, not CNFforecast,temporalvalidation,DErankguarantee,officialscore or reward. Entirely stages<=8.25. No future expression or future-score-driven fitting. No independent embryo identity.',
          'decision':'Proceed to predeclared temporal scoring only if all4 observation cases converge with finite nonnegative locked-support outputs; failure retains residuals and does not exhaust other structural methods.'}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2));events=out/'events.jsonl'
    append_event(events,'plan_frozen',sha256=digest(out/'plan.json'))
    validate_controls(); append_event(events,'unit_controls_passed',controls_sha256=plan['controls_sha256'])
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
        projected,audit=solve(candidate,reference,iterations=plan['iterations'])
        result={'cutoff':cutoff,'held_capture':held,'current_cm_cells':len(rows),'source_cm_cells':len(other),
                'mapped_genes':len(atlas),'added_detections':added_total,'removed_detections':removed_total,
                'finite_nonnegative':bool(np.isfinite(projected).all() and np.all(projected>=0)),
                'projection':audit,'rows_sha256':digest(out/f'rows_{cutoff}_{held}.npy')}
        results.append(result);append_event(events,'case_completed',**result)
        del reference,candidate,projected
    report={'status':'completed','plan':plan,'results':results,'all_cases_viable':all(r['projection']['valid'] and r['finite_nonnegative'] for r in results),
            'official_score':None,'headline_score':None,'raw_metrics':None,'skills':None,'reward_delta':0}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    public={**report,'plan_sha256':digest(out/'plan.json'),'report_sha256':digest(out/'report.json')}
    (HERE/'JOINT_MARGIN_SOLVER_RESULTS.json').write_text(json.dumps(public,indent=2))
    append_event(events,'audit_completed',all_cases_viable=report['all_cases_viable'])


if __name__=='__main__':
    try:
        with threadpool_limits(limits=2):main()
    except Exception as exc:
        out=HERE/'private'/RUN
        if out.exists():append_event(out/'events.jsonl','run_failed',exception_type=type(exc).__name__,message=str(exc))
        raise
