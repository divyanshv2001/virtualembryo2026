"""Past-only necessary bounds for mean/norm matching of zero-locked donor controls."""
import json
from collections import Counter
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now,append_event
from lineage_residual_screen import lineage
from projection_survival_diagnostics import PastReadGuard
from capture_mean_decoder_fullpanel import decode
from temporary_forecast_cache import TemporaryForecastCache
from cnf_manifold_flow import DensityFlowNet
from partial_anchor_forecast import PartialAnchorForecast
from robust_population import covariance_change
from zero_creation_blocked_decoder import decode_blocked
from supported_mean_energy import project,numerical_controls
from forecast_transform_fidelity import implied_mass

RUN='supported_mean_energy_preflight_01'
PUBLIC='SUPPORTED_MEAN_ENERGY_PREFLIGHT_RESULTS.json'


def main():
    out=HERE/'private'/RUN
    if out.exists():raise ValueError('Preserve unique frozen run')
    spec_path=HERE/'NEXT_SUPPORTED_MEAN_ENERGY_PREFLIGHT.json';spec=json.loads(spec_path.read_text());source=HERE/'private/associated_prepared_01';prepared=json.loads((source/'report.json').read_text())
    for name,key in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(source/name)!=prepared[key]:raise ValueError('Source changed')
    previous=HERE/'private/zero_creation_blocked_fullpanel_01';prior=json.loads((previous/'report.json').read_text());mean_root=HERE/'private/capture_mean_shrinkage_audit_01';means=json.loads((mean_root/'report.json').read_text())
    m=pd.read_csv(source/'selected_metadata.csv');stages=m.numeric_stage.to_numpy(float);labels=m.celltype_extended_atlas.map(lineage).to_numpy();symbols=pd.read_csv(source/'genes.csv').symbol.fillna('').tolist();freq=Counter(symbols)
    panel_path=HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt';panel=panel_path.read_text().splitlines();lookup={s:i for i,s in enumerate(panel)}
    columns=np.array([i for i,s in enumerate(symbols) if s in lookup and freq[s]==1]);mapped=np.array([lookup[symbols[i]] for i in columns]);x=np.load(source/'expression.npy',mmap_mode='r')
    plan={**spec,'created_utc':now(),'spec_sha256':digest(spec_path),'prepared_report_sha256':digest(source/'report.json'),'prior_report_sha256':digest(previous/'report.json'),'mean_report_sha256':digest(mean_root/'report.json'),'panel_sha256':digest(panel_path),
          'code_sha256':{n:digest(HERE/n) for n in ['supported_mean_energy_preflight.py','supported_mean_energy.py','zero_creation_blocked_decoder.py','capture_mean_decoder_fullpanel.py','partial_anchor_forecast.py','log1p_positive_forecast.py','temporary_forecast_cache.py','projection_survival_diagnostics.py','lineage_residual_screen.py','robust_population.py']}}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2));(out/'executed_source.py').write_bytes((HERE/'supported_mean_energy_preflight.py').read_bytes());events=out/'events.jsonl';append_event(events,'plan_frozen',sha256=digest(out/'plan.json'));numerical_controls();append_event(events,'numerical_controls_passed');results=[]
    for fold in prior['folds']:
        cutoff,target=fold['cutoff'],fold['target'];guard=PastReadGuard(x,stages,cutoff);donor_path=previous/f'cutoff_{cutoff}'/'donor_rows.npy'
        if digest(donor_path)!=fold['donor_rows_sha256']:raise ValueError('Donor split changed')
        rows=np.load(donor_path);donors=np.zeros((len(rows),len(panel)),np.float32)
        for start in range(0,len(columns),256):donors[:,mapped[start:start+256]]=np.asarray(guard[np.ix_(rows,columns[start:start+256])],np.float32)
        arc=HERE/f'private/cnf_hurdle_temporal_01/cutoff_{cutoff}'
        for n,sha in prior['plan']['archive_sha256'][str(cutoff)].items():
            if digest(arc/n)!=sha:raise ValueError('Archive changed')
        with np.load(arc/'encoder.npz') as enc:
            features=enc['features'];cov_features=enc['guard_features'];center=enc['center'];scale=enc['scale'];net=DensityFlowNet(enc['basis'],enc['pca_center'],cutoff,7.25)
        net.load_state_dict(torch.load(arc/'training.pt',weights_only=False,map_location='cpu')['net']);net.eval();program=PartialAnchorForecast(guard,stages,cutoff,donors,panel,symbols,net,center,scale,features,cov_features);program.configure(.5,.5)
        case=next(c for c in means['fit_cases'] if c['cutoff']==cutoff);mean_path=mean_root/f'cutoff_{cutoff}'/'means.npz'
        if digest(mean_path)!=case['forecast_sha256']:raise ValueError('Mean forecasts changed')
        supported=[r['lineage'] for r in case['records'] if r['status']=='supported']
        with np.load(mean_path) as saved:
            if not np.array_equal(columns,saved['columns']):raise ValueError('Mapping changed')
            for name in spec['candidates']:
                if name=='copy':pred=donors.copy()
                elif name=='anchor_unshrunk':pred,ids,audit=program.predict(target,'joint',1.,sampling='systematic')
                elif name.startswith('blocked_'):pred=decode_blocked(donors,mapped,labels[rows],saved,supported,name.replace('blocked_','capture_'))
                else:pred=decode(donors,mapped,labels[rows],saved,supported,name)
                cache=TemporaryForecastCache(HERE/'private/temporary_cache')
                try:sha=cache.put(name,pred)
                finally:cache.close()
                expected=next(r for r in fold['results'] if r['candidate']==name)['prediction_sha256']
                if sha!=expected:raise ValueError('Exact scored forecast replay failed')
                result={'cutoff':cutoff,'candidate':name,'prediction_sha256':sha,'exact_replay_passed':True,'max_fit_stage':guard.max_stage,'role':'replayed_control','headline_score':None,'raw_metrics':None,'skills':None,'reward_delta':0}
                if name in ['capture_quarter','capture_shrunk']:
                    projected,records=project(donors,pred,mapped,labels[rows],supported)
                    blocked=decode_blocked(donors,mapped,labels[rows],saved,supported,name)
                    distance=0.;blocked_energy=0.
                    for start in range(0,len(mapped),256):
                        cols=mapped[start:start+256];v=np.asarray(projected[:,cols],float);b=np.asarray(blocked[:,cols],float);a=np.asarray(donors[:,cols],float)
                        distance+=float(np.sum((v-b)**2));blocked_energy+=float(np.sum((b-a)**2))
                    protected=np.setdiff1d(np.arange(len(panel)),mapped);fallback=~np.isin(labels[rows],supported)
                    mass=implied_mass(donors,mapped);mass_error=float(np.max(np.abs(implied_mass(projected,mapped)-mass)/np.maximum(mass,1e-9)))
                    mean_change=float(np.max(np.abs(projected.mean(0,dtype=float)-donors.mean(0,dtype=float))))
                    covariance=covariance_change(donors[:,cov_features],projected[:,cov_features])
                    zero_lock=bool(np.all(projected[donors==0]==0));finite=bool(np.isfinite(projected).all() and np.all(projected>=0))
                    unchanged=bool(np.array_equal(projected[:,protected],donors[:,protected]));fallback_ok=bool(np.array_equal(projected[fallback],donors[fallback]))
                    cache=TemporaryForecastCache(HERE/'private/temporary_cache')
                    try:projected_sha=cache.put('projected_'+name,projected)
                    finally:cache.close()
                    guards=bool(finite and zero_lock and unchanged and fallback_ok and mass_error<=1e-5 and mean_change<=.5 and covariance<=.4)
                    candidate={'cutoff':cutoff,'candidate':'projected_'+name,'prediction_sha256':projected_sha,'source_control_sha256':sha,'max_fit_stage':guard.max_stage,'role':'new_preflight',
                               'lineages':records,'zero_lock':zero_lock,'finite_nonnegative':finite,'protected_genes_unchanged':unchanged,'fallback_unchanged':fallback_ok,
                               'count_mass_relative_error':mass_error,'max_mean_change':mean_change,'covariance_change':covariance,'guards_passed':guards,
                               'all_lineage_energy_budgets_passed':all(r['energy_budget_passed'] for r in records),'squared_distance_from_blocked':distance,
                               'distance_relative_to_blocked_displacement':float((distance/blocked_energy)**.5) if blocked_energy else None,
                               'headline_score':None,'raw_metrics':None,'skills':None,'reward_delta':0}
                    results.append(candidate);append_event(events,'projection_checked',cutoff=cutoff,candidate=candidate['candidate'],guards_passed=guards,energy_budgets_passed=candidate['all_lineage_energy_budgets_passed'])
                    del projected,blocked
                results.append(result);append_event(events,'control_replayed',cutoff=cutoff,candidate=name);del pred
        del program,net,donors
    report={'status':'completed','plan':plan,'results':results,'target_expression_read':False,'full_panel_scoring_executed':False,'reward_delta':0}
    (out/'report.json').write_text(json.dumps(report,indent=2));(HERE/PUBLIC).write_text(json.dumps({**report,'report_sha256':digest(out/'report.json')},indent=2));append_event(events,'audit_completed',report_sha256=digest(out/'report.json'))


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
