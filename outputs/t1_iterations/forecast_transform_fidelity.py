"""Donor/past-fit-only transform audit; exact old forecast replay, no target read."""
import json
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event
from robust_population import covariance_change
from temporary_forecast_cache import TemporaryForecastCache


def implied_mass(matrix, columns):
    mass=np.zeros(len(matrix),dtype=np.float64)
    for start in range(0,len(columns),256):
        mass+=np.expm1(np.asarray(matrix[:,columns[start:start+256]],dtype=np.float64)).sum(1)
    return mass


def main():
    out=HERE/'private/forecast_transform_fidelity_01'
    if out.exists():raise ValueError('Preserve original audit')
    source=HERE/'private/associated_prepared_01'
    manifest=json.loads((source/'report.json').read_text())
    for name,key in [('expression.npy','expression_sha256'),('genes.csv','genes_sha256'),('selected_metadata.csv','metadata_sha256')]:
        if digest(source/name)!=manifest[key]:raise ValueError('Prepared inputs changed')
    public_path=HERE/'SOURCE_SLOPE_BACKOFF_RESULTS.json'
    prior=json.loads(public_path.read_text())
    panel_path=HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt'
    panel=panel_path.read_text().splitlines();symbols=pd.read_csv(source/'genes.csv').symbol.fillna('').tolist()
    frequency=Counter(symbols);lookup={s:i for i,s in enumerate(symbols) if s and frequency[s]==1}
    mapped=np.array([i for i,s in enumerate(panel) if s in lookup]);columns=np.array([lookup[panel[i]] for i in mapped])
    plan={'created_utc':now(),'code_sha256':digest(HERE/'forecast_transform_fidelity.py'),
        'dependencies_sha256':{n:digest(HERE/n) for n in ['train_extended_atlas.py','temporary_forecast_cache.py','robust_population.py']},
        'source_report_sha256':digest(source/'report.json'),'prior_report_sha256':digest(public_path),
        'panel_sha256':digest(panel_path),'folds':[[f['cutoff'],f['target']] for f in prior['folds']],
        'transforms':['nonnegative_clip_only','mapped_log_sum_repair','mapped_expm1_sum_repair'],
        'alpha':.125,'horizon_steps':4,'seed':20260928,
        'scope':'Donors and previously frozen past slope coefficients only. No target matrices, scoring, model fit, reward or promotion. Full panel unmapped columns stay unchanged.',
        'resources':'D-only temporary single-forecast cache; implied-mass256gene streaming and source memmap; no retained full forecasts.'}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (out/'executed_source.py').write_bytes((HERE/'forecast_transform_fidelity.py').read_bytes())
    events=out/'events.jsonl';append_event(events,'plan_frozen',plan_sha256=digest(out/'plan.json'))
    x=np.load(source/'expression.npy',mmap_mode='r');rows=[]
    for fold in prior['folds']:
        cutoff=fold['cutoff'];folder=HERE/'private/source_slope_fullpanel_02'/f'cutoff_{cutoff}'
        donor_path=folder/'donor_rows.npy'
        if digest(donor_path)!=fold['donor_rows_sha256']:raise ValueError('Donor split changed')
        donor_rows=np.load(donor_path)
        donors=np.zeros((len(donor_rows),len(panel)),dtype=np.float32)
        donors[:,mapped]=np.asarray(x[np.ix_(donor_rows,columns)],dtype=np.float32)
        reference=donors[:,mapped].mean(0,dtype=np.float64)
        donor_logmass=donors[:,mapped].sum(1,dtype=np.float64)
        donor_countmass=implied_mass(donors,mapped)
        with np.load(HERE/'private/cnf_hurdle_temporal_01'/f'cutoff_{cutoff}'/'encoder.npz') as encoder:guard=encoder['guard_features']
        slope_path=HERE/'private/lineage_time_partial_pool_proxy_01'/f'deltas_{cutoff}.npz'
        expected=fold['generation']['slope_recent']['audit']['slope_artifact_sha256']
        if digest(slope_path)!=expected:raise ValueError('Slope artifact changed')
        with np.load(slope_path) as saved:
            if not np.array_equal(mapped,saved['mapped']):raise ValueError('Mapped panel changed')
            slopes={name:saved[key].copy() for name,key in [('slope_recent','recent_linear'),('slope_global','global'),('slope_lineage1000','lineage_1000.0')]}
        for name,slope in slopes.items():
            intended=4*.125*slope;mask=np.abs(intended)>1e-6
            for method in plan['transforms']:
                pred=donors.copy()
                values=pred[:,mapped]+intended
                clipped=int((values<0).sum());pred[:,mapped]=np.maximum(0,values);del values
                if method=='mapped_log_sum_repair':
                    changed=pred[:,mapped].sum(1,dtype=np.float64)
                    factor=np.divide(donor_logmass,changed,out=np.ones_like(changed),where=changed>0)
                    pred[:,mapped]*=factor[:,None]
                elif method=='mapped_expm1_sum_repair':
                    changed=implied_mass(pred,mapped)
                    factor=np.divide(donor_countmass,changed,out=np.ones_like(changed),where=changed>0)
                    abundance=np.expm1(np.asarray(pred[:,mapped],dtype=np.float64))
                    pred[:,mapped]=np.log1p(abundance*factor[:,None]).astype(np.float32);del abundance
                realized=pred[:,mapped].mean(0,dtype=np.float64)-reference
                countmass=implied_mass(pred,mapped);logmass=pred[:,mapped].sum(1,dtype=np.float64)
                cache=TemporaryForecastCache(HERE/'private/temporary_cache')
                try:forecast_hash=cache.put('diagnostic',pred)
                finally:cache.close()
                if method=='mapped_log_sum_repair' and forecast_hash!=fold['generation'][name]['prediction_sha256']:
                    raise ValueError('Exact prior log-repair forecast replay failed')
                mean_change=float(np.max(np.abs(realized)));covariance=covariance_change(donors[:,guard],pred[:,guard])
                row={'cutoff':cutoff,'candidate':name,'transform':method,'prediction_sha256':forecast_hash,
                    'intended_realized_spearman':float(spearmanr(intended,realized).statistic),
                    'material_delta_sign_agreement':float((np.sign(intended[mask])==np.sign(realized[mask])).mean()),
                    'relative_delta_l2_error':float(np.linalg.norm(realized-intended)/max(np.linalg.norm(intended),1e-12)),
                    'realized_reference_magnitude_spearman':float(spearmanr(realized,reference).statistic),
                    'negative_entries_clipped':clipped,'maximum_mean_perturbation':mean_change,
                    'covariance_change':covariance,'same_mean_covariance_guard_pass':bool(mean_change<=.5 and covariance<=.4),
                    'mapped_implied_count_mass_median_relative_change':float(np.median(countmass/np.maximum(donor_countmass,1e-9)-1)),
                    'mapped_implied_count_mass_max_relative_error':float(np.max(np.abs(countmass-donor_countmass)/np.maximum(donor_countmass,1e-9))),
                    'mapped_log_mass_max_relative_error':float(np.max(np.abs(logmass-donor_logmass)/np.maximum(donor_logmass,1e-9))),
                    'donor_mapped_implied_count_mass_median':float(np.median(donor_countmass))}
                rows.append(row);append_event(events,'transform_audited',**row);del pred
    report={'updated_utc':now(),'plan_sha256':digest(out/'plan.json'),'outcomes':rows,
        'exact_prior_log_repair_forecasts_replayed':6,'source_normalization_target':manifest['normalization_target'],
        'normalization_interpretation':'Sum of log1p values is not total normalized count mass. Full atlas normalization targets10000 implied counts; mapped subset excludes some genes. Count repair here preserves each observed mapped subset total, not unknown genes.',
        'models_fit':0,'full_panel_scores':0,'reward_delta':0,'target_data_read':False,
        'next_action':'Declare matched mapped-count-space repair ablation with frozen donor-only backoffs; hold original log-repair outcomes and control hashes unchanged. This audit cannot certify forecast improvement.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    public=dict(report,report_sha256=digest(out/'report.json'))
    (HERE/'FORECAST_TRANSFORM_FIDELITY_RESULTS.json').write_text(json.dumps(public,indent=2)+'\n')
    append_event(events,'audit_completed',report_sha256=public['report_sha256'])
    print(json.dumps({k:v for k,v in public.items() if k!='outcomes'}))


if __name__=='__main__':main()
