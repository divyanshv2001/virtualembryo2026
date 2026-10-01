"""Past-only decoded-energy matching of geometric and uniform lineage controls."""
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
from capture_ols_gate_decoder import decode as gate_decode
from capture_amplitude_matched import shifts,controls
from forecast_transform_fidelity import implied_mass
from repair_aware_energy import attenuate,numerical_controls

RUN='capture_decoded_energy_retry_preflight_01'
PUBLIC='CAPTURE_DECODED_ENERGY_RETRY_PREFLIGHT_RESULTS.json'


def main():
    out=HERE/'private'/RUN
    if out.exists():raise ValueError('Preserve unique frozen run')
    spec_path=HERE/'NEXT_CAPTURE_DECODED_ENERGY_RETRY_PREFLIGHT.json';spec=json.loads(spec_path.read_text());source=HERE/'private/associated_prepared_01';prepared=json.loads((source/'report.json').read_text())
    for name,key in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(source/name)!=prepared[key]:raise ValueError('Source changed')
    previous=HERE/'private/zero_creation_blocked_fullpanel_01';prior=json.loads((previous/'report.json').read_text());mean_root=HERE/'private/capture_module_stability_audit_01';means=json.loads((mean_root/'report.json').read_text());matched_root=HERE/'private/matched_capture_module_OLS_omission_audit_01';matched=json.loads((matched_root/'report.json').read_text())
    if digest(matched_root/'report.json')!=spec['source_report_sha256']:raise ValueError('Matched source changed')
    if digest(HERE/'private/capture_OLS_decoder_boundary_audit_01/report.json')!=spec['boundary_report_sha256']:raise ValueError('Boundary source changed')
    if digest(HERE/'private/capture_geometric_rate_preflight_01/report.json')!=spec['geometric_report_sha256']:raise ValueError('Geometric source changed')
    amplitude_path=HERE/'private/capture_amplitude_matched_preflight_01/report.json'
    if digest(amplitude_path)!=spec['amplitude_report_sha256']:raise ValueError('Amplitude report changed')
    amplitude=json.loads(amplitude_path.read_text())
    m=pd.read_csv(source/'selected_metadata.csv');stages=m.numeric_stage.to_numpy(float);labels=m.celltype_extended_atlas.map(lineage).to_numpy();symbols=pd.read_csv(source/'genes.csv').symbol.fillna('').tolist();freq=Counter(symbols)
    panel_path=HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt';panel=panel_path.read_text().splitlines();lookup={s:i for i,s in enumerate(panel)}
    columns=np.array([i for i,s in enumerate(symbols) if s in lookup and freq[s]==1]);mapped=np.array([lookup[symbols[i]] for i in columns]);x=np.load(source/'expression.npy',mmap_mode='r')
    plan={**spec,'created_utc':now(),'spec_sha256':digest(spec_path),'prepared_report_sha256':digest(source/'report.json'),'prior_report_sha256':digest(previous/'report.json'),'mean_report_sha256':digest(mean_root/'report.json'),'panel_sha256':digest(panel_path),
          'code_sha256':{n:digest(HERE/n) for n in ['capture_decoded_energy_retry_preflight.py','capture_ols_gate_decoder.py','capture_geometric_rate.py','capture_amplitude_matched.py','repair_aware_energy.py','zero_creation_blocked_decoder.py','capture_mean_decoder_fullpanel.py','partial_anchor_forecast.py','log1p_positive_forecast.py','temporary_forecast_cache.py','projection_survival_diagnostics.py','lineage_residual_screen.py','robust_population.py']}}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2));(out/'executed_source.py').write_bytes((HERE/'capture_decoded_energy_retry_preflight.py').read_bytes());events=out/'events.jsonl';append_event(events,'plan_frozen',sha256=digest(out/'plan.json'));controls();numerical_controls();append_event(events,'numerical_controls_passed');results=[]
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
        supported=[r['lineage'] for r in means['results'] if r['cutoff']==cutoff and r['every_omission_supported']]
        all_shifts={};source_artifacts={}
        for group in supported:
            mean_case=next(r for r in means['results'] if r['cutoff']==cutoff and r['lineage']==group)
            matched_case=next(r for r in matched['results'] if r['cutoff']==cutoff and r['lineage']==group)
            mean_path=mean_root/f'cutoff_{cutoff}'/(group+'.npz');matched_path=matched_root/f'cutoff_{cutoff}'/(group+'.npz')
            if digest(mean_path)!=mean_case['artifact_sha256'] or digest(matched_path)!=matched_case['matched_artifact_sha256']:raise ValueError('Frozen artifact changed')
            with np.load(mean_path) as original,np.load(matched_path) as saved:
                if not np.array_equal(columns,original['columns']) or not np.array_equal(original['projected'],saved['projected']) or not np.array_equal(original['unprojected_ols'],saved['unprojected_OLS']):raise ValueError('Source array mapping changed')
                all_shifts[group]=shifts(original['capture_means'],original['capture_stages'],saved['unprojected_OLS'],saved['projected'],saved['omitted_OLS'])
            source_artifacts[group]={'means_sha256':digest(mean_path),'matched_sha256':digest(matched_path)}
        protected=np.setdiff1d(np.arange(len(panel)),mapped);fallback=~np.isin(labels[rows],supported);mass=implied_mass(donors,mapped)
        originals=['capture_geometric_decay','capture_OLS_decay_support_amplitude_matched','capture_recent_decay_support_amplitude_matched']
        budgets=[{'lineage':g,'frozen_energy_budget':min(next(f['decoded_cell_shift_energy'] for f in r['shift_fidelity'] if f['lineage']==g) for r in amplitude['results'] if r['cutoff']==cutoff and r['candidate'] in originals)} for g in supported]
        for name in spec['candidates']:
            base=name.removesuffix('_decoded_matched');energy_matching=[];clipped=0
            if name=='copy':pred=donors.copy()
            elif name=='anchor_unshrunk':pred,ids,audit=program.predict(target,'joint',1.,sampling='systematic')
            else:pred,clipped=gate_decode(donors,mapped,labels[rows],{g:all_shifts[g][base] for g in supported})
            if name.endswith('_decoded_matched'):
                pred,energy_matching=attenuate(donors,pred,mapped,labels[rows],budgets)
                for record in energy_matching:
                    record['energy_match_error']=abs(record['final_energy']-record['frozen_energy_budget'])
                    record['energy_equality_passed']=record['energy_match_error']<=1e-6*max(1.,record['frozen_energy_budget'])
            cache=TemporaryForecastCache(HERE/'private/temporary_cache')
            try:sha=cache.put(name,pred)
            finally:cache.close()
            if not name.endswith('_decoded_matched'):
                expected=next(r for r in amplitude['results'] if r['cutoff']==cutoff and r['candidate']==name)['prediction_sha256']
                if sha!=expected:raise ValueError('Original amplitude forecast replay changed')
            replay=None
            if name in ['copy','anchor_unshrunk']:
                expected=next(r for r in fold['results'] if r['candidate']==name)['prediction_sha256']
                if sha!=expected:raise ValueError('Exact scored control replay failed')
                replay=True
            if name=='capture_OLS':
                old=json.loads((HERE/'private/capture_OLS_gate_decoder_preflight_01/report.json').read_text())
                expected=next(r for r in old['results'] if r['cutoff']==cutoff and r['candidate']==name)['prediction_sha256']
                if sha!=expected:raise ValueError('Original OLS replay changed')
                replay=True
            if name=='capture_geometric_decay':
                original=json.loads((HERE/'private/capture_geometric_rate_preflight_01/report.json').read_text())
                expected=next(r for r in original['results'] if r['cutoff']==cutoff and r['candidate']==name)['prediction_sha256']
                if sha!=expected:raise ValueError('Geometric forecast replay changed')
                replay=True
            fidelity=[]
            if name not in ['copy','anchor_unshrunk']:
                for group in supported:
                    ids=np.flatnonzero(labels[rows]==group);shift=all_shifts[group][base];actual=pred[ids][:,mapped].mean(0,dtype=float)-donors[ids][:,mapped].mean(0,dtype=float);norm=float(np.linalg.norm(shift));ols_reference=all_shifts[group]['capture_OLS']
                    fidelity.append({'lineage':group,'shift_norm':norm,'decoded_norm':float(np.linalg.norm(actual)),
                        'relative_shift_error':float(np.linalg.norm(actual-shift)/norm) if norm else None,
                        'retained_OLS_shift_energy':float(shift@shift/(ols_reference@ols_reference)) if ols_reference@ols_reference else None,
                        'nonzero_shift_genes':int(np.count_nonzero(shift)),
                        'requested_cell_shift_energy':float(len(ids)*(shift@shift)),
                        'decoded_cell_shift_energy':sum(float(np.square(np.asarray(pred[np.ix_(ids,mapped[j:j+256])],float)-np.asarray(donors[np.ix_(ids,mapped[j:j+256])],float)).sum()) for j in range(0,len(mapped),256)),
                        'decoded_gene_sign_reversals':int(((shift!=0)&(np.abs(actual)>1e-8)&(np.sign(actual)!=np.sign(shift))).sum())})
            mass_error=float(np.max(np.abs(implied_mass(pred,mapped)-mass)/np.maximum(mass,1e-9)))
            mean_change=float(np.max(np.abs(pred.mean(0,dtype=float)-donors.mean(0,dtype=float))));covariance=covariance_change(donors[:,cov_features],pred[:,cov_features])
            unchanged=bool(np.array_equal(pred[:,protected],donors[:,protected]));fallback_ok=bool(np.array_equal(pred[fallback],donors[fallback])) if name!='anchor_unshrunk' else None
            finite=bool(np.isfinite(pred).all() and np.all(pred>=0))
            guards=bool(finite and unchanged and (fallback_ok or name=='anchor_unshrunk') and mass_error<=1e-5 and mean_change<=.5 and covariance<=.4)
            result={'cutoff':cutoff,'candidate':name,'prediction_sha256':sha,'exact_control_replay':replay,'max_fit_stage':guard.max_stage,'source_artifacts':source_artifacts,
                'supported_lineages':supported,'clipped_entries':clipped,'count_mass_relative_error':mass_error,'max_mean_change':mean_change,'covariance_change':covariance,
                'finite_nonnegative':finite,'protected_genes_unchanged':unchanged,'fallback_unchanged':fallback_ok,'guards_passed':guards,'shift_fidelity':fidelity,
                'decoded_energy_matching':energy_matching,'all_decoded_energy_matches_passed':all(r['energy_equality_passed'] and r['final_energy_budget_passed'] for r in energy_matching) if energy_matching else None,
                'headline_score':None,'raw_metrics':None,'skills':None,'reward_delta':0}
            results.append(result);append_event(events,'decoder_checked',cutoff=cutoff,candidate=name,guards_passed=guards);del pred
        del program,net,donors
    report={'status':'completed','plan':plan,'results':results,'target_expression_read':False,'full_panel_scoring_executed':False,'reward_delta':0}
    (out/'report.json').write_text(json.dumps(report,indent=2));(HERE/PUBLIC).write_text(json.dumps({**report,'report_sha256':digest(out/'report.json')},indent=2));append_event(events,'audit_completed',report_sha256=digest(out/'report.json'))


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
