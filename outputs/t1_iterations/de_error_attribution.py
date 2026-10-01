"""Posthoc diagnostic only; never supplies target evidence to forecast fitting."""
import json
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event
from forecast_transform_fidelity import implied_mass
from offline_backtest import load_core
from lineage_residual_screen import lineage
from temporary_forecast_cache import TemporaryForecastCache


def rho(a,b):
    if np.std(a)<1e-12 or np.std(b)<1e-12:return None
    return float(spearmanr(a,b).statistic)


def main():
    out=HERE/'private/de_error_attribution_01'
    if out.exists():raise ValueError('Preserve diagnostic evidence')
    previous_path=HERE/'SOURCE_SLOPE_TRANSFORM_RESULTS.json'
    previous=json.loads(previous_path.read_text())
    source=HERE/'private/associated_prepared_01'
    plan={'created_utc':now(),'code_sha256':digest(HERE/'de_error_attribution.py'),
          'prior_report_sha256':digest(previous_path),'folds':[[f['cutoff'],f['target']] for f in previous['folds']],
          'seed':20260928,'scope':'Labeled posthoc diagnosis on reused-development targets. No fit, candidate/backoff selection, reward or new benchmark scores. Do not reuse targets for promotion of any resulting method.',
          'methods':['slope_recent','slope_clip','slope_count'],
          'dependencies_sha256':{n:digest(HERE/n) for n in ['forecast_transform_fidelity.py','offline_backtest.py','lineage_residual_screen.py','temporary_forecast_cache.py']}}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (out/'executed_source.py').write_bytes((HERE/'de_error_attribution.py').read_bytes())
    events=out/'events.jsonl';append_event(events,'plan_frozen',sha256=digest(out/'plan.json'))
    prepared=json.loads((source/'report.json').read_text())
    for filename,key in [('expression.npy','expression_sha256'),('genes.csv','genes_sha256'),('selected_metadata.csv','metadata_sha256')]:
        if digest(source/filename)!=prepared[key]:raise ValueError('Prepared source changed')
    x=np.load(source/'expression.npy',mmap_mode='r')
    metadata=pd.read_csv(source/'selected_metadata.csv');labels=metadata.celltype_extended_atlas.map(lineage).to_numpy()
    symbols=pd.read_csv(source/'genes.csv').symbol.fillna('').tolist();counts=Counter(symbols)
    lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
    panel=(HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
    mapped=np.array([i for i,s in enumerate(panel) if s in lookup]);cols=np.array([lookup[panel[i]] for i in mapped])
    def values(rows):
        a=np.zeros((len(rows),len(panel)),dtype=np.float32);a[:,mapped]=np.asarray(x[np.ix_(rows,cols)]);return a
    core,_=load_core();results=[]
    for fold in previous['folds']:
        cutoff=fold['cutoff'];folder=HERE/'private/source_slope_transform_01'/f'cutoff_{cutoff}'
        donor_path=folder/'donor_rows.npy';target_path=folder/'target_rows.npy'
        if digest(donor_path)!=fold['donor_rows_sha256'] or digest(target_path)!=fold['target_rows_sha256']:raise ValueError('Split changed')
        donor_rows=np.load(donor_path);donors=values(donor_rows)
        slope_path=HERE/'private/lineage_time_partial_pool_proxy_01'/f'deltas_{cutoff}.npz'
        if digest(slope_path)!=fold['generation']['slope_recent']['audit']['slope_artifact_sha256']:raise ValueError('Past fit changed')
        with np.load(slope_path) as saved:
            if not np.array_equal(saved['mapped'],mapped):raise ValueError('Mapping changed')
            slope=saved['recent_linear'].copy()
        dp_bank={};hashes={};ref=donors.mean(0);ref_detection=(donors>0).mean(0)
        for method in plan['methods']:
            alpha=fold['generation'][method]['audit']['selected_shrinkage']
            pred=donors.copy();pred[:,mapped]=np.maximum(0,pred[:,mapped]+4*alpha*slope)
            if method=='slope_recent':
                original=donors[:,mapped].sum(1,dtype=np.float64);changed=pred[:,mapped].sum(1,dtype=np.float64)
                pred[:,mapped]*=np.divide(original,changed,out=np.ones_like(original),where=changed>0)[:,None]
            elif method=='slope_count':
                original=implied_mass(donors,mapped);changed=implied_mass(pred,mapped)
                factor=np.divide(original,changed,out=np.ones_like(original),where=changed>0)
                for start in range(0,len(mapped),256):
                    c=mapped[start:start+256];pred[:,c]=np.log1p(np.expm1(np.asarray(pred[:,c],dtype=np.float64))*factor[:,None]).astype(np.float32)
            cache=TemporaryForecastCache(HERE/'private/temporary_cache')
            try:sha=cache.put(method,pred)
            finally:cache.close()
            if sha!=fold['generation'][method]['prediction_sha256']:raise ValueError('Forecast replay failed: '+method)
            hashes[method]=sha;dp_bank[method]=pred.mean(0)-ref;del pred
        append_event(events,'forecasts_replayed_before_diagnostic_target_read',cutoff=cutoff,hashes=hashes)
        target_rows=np.load(target_path);order=np.random.default_rng(plan['seed']).permutation(len(target_rows));truth_rows=target_rows[order[:1000]]
        truth=values(truth_rows);up,down,dt=core.de_genes(truth,donors)
        true_set=np.concatenate([up,down]);chance=max(core._signed_overlap(ref,up,down)[0],core._signed_overlap(-ref,up,down)[0])
        threshold=float(core.MIN_LFC);effect_set=np.flatnonzero(np.abs(dt)>=threshold)
        rl=labels[donor_rows];tl=labels[truth_rows];groups=sorted(set(rl)|set(tl))
        composition=np.zeros(len(panel));within=np.zeros(len(panel));group_rows=[];covered_r=covered_t=0.
        intended=np.zeros(len(panel));intended[mapped]=slope
        for group in groups:
            rmask=rl==group;tmask=tl==group;nr=int(rmask.sum());nt=int(tmask.sum());pr=nr/len(donors);pt=nt/len(truth)
            row={'lineage':group,'reference_cells':nr,'truth_cells':nt,'reference_fraction':pr,'truth_fraction':pt,'fraction_change':pt-pr}
            if nr>=20 and nt>=20:
                rm=donors[rmask].mean(0,dtype=np.float64);tm=truth[tmask].mean(0,dtype=np.float64)
                composition+=(pt-pr)*rm;within+=pt*(tm-rm);covered_r+=pr;covered_t+=pt
                row['intended_within_lineage_direction_rho']=rho(intended,tm-rm)
            group_rows.append(row)
        rows=[]
        for method,dp in dp_bank.items():
            raw,n=core._signed_overlap(dp,up,down);std_ratio=float(np.std(dp)/max(np.std(dt),1e-12))
            score=0. if std_ratio<.01 else float((raw-chance)/(1-chance))
            old=next(v for v in fold['results'] if v['candidate']==method)
            if not np.isclose(score,old['raw_metrics']['de_score'],atol=1e-10):raise ValueError('DE scorer replay mismatch')
            signed_order=np.argsort(-np.asarray(dp,float));pu=signed_order[:len(up)];pdn=signed_order[len(signed_order)-len(down):] if len(down) else np.array([],int)
            row={'candidate':method,'prediction_sha256':hashes[method],'de_score_replayed':score,'signed_overlap':raw,
                 'up_hits':int(len(np.intersect1d(pu,up))),'down_hits':int(len(np.intersect1d(pdn,down))),
                 'truth_de_sign_agreement':float((np.sign(dp[true_set])==np.sign(dt[true_set])).mean()),
                 'whole_panel_direction_rho':rho(dp,dt),'truth_de_direction_rho':rho(dp[true_set],dt[true_set]),
                 'direction_rho_composition_component':rho(dp,composition),'direction_rho_within_component':rho(dp,within),
                 'std_delta_ratio':std_ratio}
            rows.append(row)
        result={'cutoff':cutoff,'target':fold['target'],'truth_up':len(up),'truth_down':len(down),'truth_DE_count':len(true_set),
                'effect_only_count':len(effect_set),'BH_effect_set_equals_effect_only':bool(np.array_equal(np.sort(true_set),effect_set)),
                'expression_magnitude_null_overlap':chance,'unmapped_truth_DE_count':int(np.sum(~np.isin(true_set,mapped))),
                'truth_DE_reference_detection_median':float(np.median(ref_detection[true_set])),
                'intended_truth_direction_rho':rho(intended,dt),'intended_truth_DE_sign_agreement':float((np.sign(intended[true_set])==np.sign(dt[true_set])).mean()),
                'composition_truth_direction_rho':rho(composition,dt),'within_truth_direction_rho':rho(within,dt),
                'group_decomposition_covered_reference_fraction':covered_r,'group_decomposition_covered_truth_fraction':covered_t,
                'group_decomposition_relative_residual':float(np.linalg.norm(dt-composition-within)/max(np.linalg.norm(dt),1e-12)),
                'lineage_diagnostics':group_rows,'candidates':rows}
        results.append(result);append_event(events,'fold_diagnosed',**result);del truth,donors
    report={'updated_utc':now(),'plan_sha256':digest(out/'plan.json'),'folds':results,'scope':plan['scope'],
            'models_fit':0,'new_full_panel_scores':0,'reward_delta':0,'causal_attribution_certified':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');report['report_sha256']=digest(out/'report.json')
    (HERE/'DE_ERROR_ATTRIBUTION_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'completed':True,'report_sha256':report['report_sha256']}))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
