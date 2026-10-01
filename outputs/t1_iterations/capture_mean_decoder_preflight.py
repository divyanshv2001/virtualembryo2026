"""Past donors only: test frozen capture-mean shifts after cell/count-mass decoding."""
import json
from collections import Counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now,append_event
from lineage_residual_screen import lineage
from projection_survival_diagnostics import PastReadGuard
from forecast_transform_fidelity import implied_mass
from robust_population import covariance_change
from temporary_forecast_cache import TemporaryForecastCache

RUN='capture_mean_decoder_preflight_01'
PUBLIC='CAPTURE_MEAN_DECODER_PREFLIGHT_RESULTS.json'


def main():
    out=HERE/'private'/RUN
    if out.exists():raise ValueError('Unique frozen run required')
    spec_path=HERE/'NEXT_CAPTURE_MEAN_DECODER_PREFLIGHT.json';spec=json.loads(spec_path.read_text())
    source=HERE/'private/associated_prepared_01';prepared=json.loads((source/'report.json').read_text())
    for name,key in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(source/name)!=prepared[key]:raise ValueError('Prepared source changed')
    m=pd.read_csv(source/'selected_metadata.csv');stages=m.numeric_stage.to_numpy(float);labels=m.celltype_extended_atlas.map(lineage).to_numpy()
    symbols=pd.read_csv(source/'genes.csv').symbol.fillna('').tolist();freq=Counter(symbols)
    panel_path=HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt';panel=panel_path.read_text().splitlines()
    if len(set(panel))!=len(panel):raise ValueError('Duplicate official genes require explicit mapping')
    panel_lookup={s:i for i,s in enumerate(panel)}
    columns=np.array([i for i,s in enumerate(symbols) if s in panel_lookup and freq[s]==1]);mapped=np.array([panel_lookup[symbols[i]] for i in columns]);protected=np.setdiff1d(np.arange(len(panel)),mapped)
    x=np.load(source/'expression.npy',mmap_mode='r')
    previous=HERE/'private/capture_mean_shrinkage_audit_01';prior=json.loads((previous/'report.json').read_text())
    plan={**spec,'created_utc':now(),'spec_sha256':digest(spec_path),'panel_sha256':digest(panel_path),'prepared_report_sha256':digest(source/'report.json'),
          'prior_report_sha256':digest(previous/'report.json'),'code_sha256':{n:digest(HERE/n) for n in ['capture_mean_decoder_preflight.py','forecast_transform_fidelity.py','robust_population.py','temporary_forecast_cache.py','projection_survival_diagnostics.py','lineage_residual_screen.py']}}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2));(out/'executed_source.py').write_bytes((HERE/'capture_mean_decoder_preflight.py').read_bytes())
    events=out/'events.jsonl';append_event(events,'plan_frozen',sha256=digest(out/'plan.json'));results=[]
    for cutoff in spec['cutoffs']:
        guard=PastReadGuard(x,stages,cutoff);case=next(c for c in prior['fit_cases'] if c['cutoff']==cutoff)
        mean_path=previous/f'cutoff_{cutoff}'/'means.npz'
        if digest(mean_path)!=case['forecast_sha256']:raise ValueError('Frozen mean forecasts changed')
        donor_path=HERE/f'private/source_slope_fullpanel_02/cutoff_{cutoff}/donor_rows.npy';donor_rows=np.load(donor_path)
        if np.any(stages[donor_rows]>cutoff):raise ValueError('Nonpast donor')
        donors=np.zeros((len(donor_rows),len(panel)),dtype=np.float32)
        for start in range(0,len(columns),256):
            donors[:,mapped[start:start+256]]=np.asarray(guard[np.ix_(donor_rows,columns[start:start+256])],np.float32)
        encoder_path=HERE/f'private/cnf_hurdle_temporal_01/cutoff_{cutoff}/encoder.npz'
        with np.load(encoder_path) as encoder:cov_features=encoder['guard_features']
        donor_labels=labels[donor_rows];mass=implied_mass(donors,mapped)
        with np.load(mean_path) as saved:
            if not np.array_equal(saved['columns'],columns):raise ValueError('Gene mapping changed')
            supported=[r['lineage'] for r in case['records'] if r['status']=='supported']
            for name in spec['candidates']:
                pred=donors.copy();intended={g:(saved[g+'__'+name]-saved[g+'__persistence']) if name!='persistence' else np.zeros(len(columns)) for g in supported}
                clipping=0
                for group,shift in intended.items():
                    rows=np.flatnonzero(donor_labels==group)
                    for start in range(0,len(columns),256):
                        cols=mapped[start:start+256];block=np.asarray(pred[np.ix_(rows,cols)],float)+shift[start:start+len(cols)]
                        clipping+=int((block<0).sum());pred[np.ix_(rows,cols)]=np.maximum(block,0).astype(np.float32)
                # Repair only supported donor rows. Unsupported lineages remain bitwise copy.
                changed_rows=np.flatnonzero(np.isin(donor_labels,supported))
                changed_mass=implied_mass(pred,mapped)
                factors=np.divide(mass,changed_mass,out=np.ones_like(mass),where=changed_mass>0)
                if name!='persistence':
                    for start in range(0,len(mapped),256):
                        cols=mapped[start:start+256];a=np.expm1(np.asarray(pred[np.ix_(changed_rows,cols)],float))
                        pred[np.ix_(changed_rows,cols)]=np.log1p(a*factors[changed_rows,None]).astype(np.float32)
                fidelity=[]
                for group,shift in intended.items():
                    rows=np.flatnonzero(donor_labels==group)
                    if not len(rows):continue
                    actual=pred[rows][:,mapped].mean(0,dtype=float)-donors[rows][:,mapped].mean(0,dtype=float)
                    norm=float(np.linalg.norm(shift));fidelity.append({'lineage':group,'donors':len(rows),'intended_norm':norm,'decoded_norm':float(np.linalg.norm(actual)),
                        'relative_shift_error':float(np.linalg.norm(actual-shift)/norm) if norm>0 else None,'shift_cosine':float(actual@shift/(np.linalg.norm(actual)*norm)) if norm*np.linalg.norm(actual)>0 else None})
                mass_error=float(np.max(np.abs(implied_mass(pred,mapped)-mass)/np.maximum(mass,1e-9)))
                mean_change=float(np.max(np.abs(pred.mean(0,dtype=float)-donors.mean(0,dtype=float))))
                covariance=covariance_change(donors[:,cov_features],pred[:,cov_features]) if name!='persistence' else 0.
                unchanged=bool(np.array_equal(pred[:,protected],donors[:,protected]));fallback=bool(np.array_equal(pred[~np.isin(donor_labels,supported)],donors[~np.isin(donor_labels,supported)]))
                finite=bool(np.isfinite(pred).all() and np.all(pred>=0));identity=name!='persistence' or np.array_equal(pred,donors)
                valid=bool(finite and unchanged and fallback and identity and mass_error<=1e-5 and mean_change<=.5 and covariance<=.4)
                with _cache() as cache:
                    prediction_sha=cache.put(name,pred)
                result={'cutoff':cutoff,'candidate':name,'donor_rows_sha256':digest(donor_path),'encoder_sha256':digest(encoder_path),'mean_forecast_sha256':digest(mean_path),'prediction_sha256':prediction_sha,
                        'donors':len(donors),'max_fit_stage':guard.max_stage,'clipped_entries':clipping,'count_mass_relative_error':mass_error,'max_mean_change':mean_change,'covariance_change':covariance,
                        'protected_genes_unchanged':unchanged,'unsupported_lineages_unchanged':fallback,'finite_nonnegative':finite,'identity_passed':bool(identity),'guards_passed':valid,'lineage_shift_fidelity':fidelity,
                        'headline_score':None,'raw_metrics':None,'skills':None,'reward_delta':0}
                results.append(result);append_event(events,'decoder_diagnosed',**result);del pred
    report={'status':'completed','plan':plan,'results':results,'target_expression_read':False,'full_panel_scoring_executed':False,'reward_delta':0,'headline_score':None,'raw_metrics':None,'skills':None}
    (out/'report.json').write_text(json.dumps(report,indent=2));(HERE/PUBLIC).write_text(json.dumps({**report,'report_sha256':digest(out/'report.json')},indent=2))
    append_event(events,'preflight_completed',report_sha256=digest(out/'report.json'));print(json.dumps({'status':'completed','guards':[(r['cutoff'],r['candidate'],r['guards_passed']) for r in results]}))


from contextlib import contextmanager
@contextmanager
def _cache():
    cache=TemporaryForecastCache(HERE/'private/temporary_cache')
    try:yield cache
    finally:cache.close()


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
