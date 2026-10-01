"""Exact donor-only replay and zero/positive-origin displacement decomposition."""
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

RUN='capture_decoder_support_audit_01'
PUBLIC='CAPTURE_DECODER_SUPPORT_AUDIT_RESULTS.json'


def variograms(donors,pred,pairs):
    # Counterfactual origin masks diagnose displacement, not valid mass-preserved forecasts.
    i,j=pairs;parts={k:[] for k in ['donor','decoded','zero_origin','positive_origin']}
    for start in range(0,len(i),512):
        a,b=i[start:start+512],j[start:start+512];da,db=donors[:,a],donors[:,b];pa,pb=pred[:,a],pred[:,b]
        values={'donor':(da,db),'decoded':(pa,pb),'zero_origin':(np.where(da==0,pa,da),np.where(db==0,pb,db)),
                'positive_origin':(np.where(da>0,pa,da),np.where(db>0,pb,db))}
        for name,(x,y) in values.items():parts[name].append((np.abs(x-y)**.5).mean(0))
    return {name:np.concatenate(v) for name,v in parts.items()}


def main():
    out=HERE/'private'/RUN
    if out.exists():raise ValueError('Preserve unique frozen run')
    spec_path=HERE/'NEXT_CAPTURE_DECODER_SUPPORT_AUDIT.json';spec=json.loads(spec_path.read_text());source=HERE/'private/associated_prepared_01';prepared=json.loads((source/'report.json').read_text())
    for name,key in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(source/name)!=prepared[key]:raise ValueError('Source changed')
    previous=HERE/'private/capture_mean_decoder_fullpanel_01';prior=json.loads((previous/'report.json').read_text());mean_root=HERE/'private/capture_mean_shrinkage_audit_01';means=json.loads((mean_root/'report.json').read_text())
    m=pd.read_csv(source/'selected_metadata.csv');stages=m.numeric_stage.to_numpy(float);labels=m.celltype_extended_atlas.map(lineage).to_numpy();symbols=pd.read_csv(source/'genes.csv').symbol.fillna('').tolist();freq=Counter(symbols)
    panel_path=HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt';panel=panel_path.read_text().splitlines();lookup={s:i for i,s in enumerate(panel)}
    columns=np.array([i for i,s in enumerate(symbols) if s in lookup and freq[s]==1]);mapped=np.array([lookup[symbols[i]] for i in columns]);x=np.load(source/'expression.npy',mmap_mode='r')
    plan={**spec,'created_utc':now(),'spec_sha256':digest(spec_path),'prepared_report_sha256':digest(source/'report.json'),'prior_report_sha256':digest(previous/'report.json'),'mean_report_sha256':digest(mean_root/'report.json'),'panel_sha256':digest(panel_path),
          'code_sha256':{n:digest(HERE/n) for n in ['capture_decoder_support_audit.py','capture_mean_decoder_fullpanel.py','partial_anchor_forecast.py','log1p_positive_forecast.py','temporary_forecast_cache.py','projection_survival_diagnostics.py','lineage_residual_screen.py','robust_population.py']}}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2));(out/'executed_source.py').write_bytes((HERE/'capture_decoder_support_audit.py').read_bytes());events=out/'events.jsonl';append_event(events,'plan_frozen',sha256=digest(out/'plan.json'));results=[]
    rng=np.random.default_rng(spec['seed']);i=rng.integers(0,len(panel),20000);j=rng.integers(0,len(panel),20000);pairs=(i[i!=j],j[i!=j]);np.savez(out/'pairs.npz',i=pairs[0],j=pairs[1])
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
                else:pred=decode(donors,mapped,labels[rows],saved,supported,name)
                cache=TemporaryForecastCache(HERE/'private/temporary_cache')
                try:sha=cache.put(name,pred)
                finally:cache.close()
                expected=next(r for r in fold['results'] if r['candidate']==name)['prediction_sha256']
                if sha!=expected:raise ValueError('Exact scored forecast replay failed')
                counts=[];zero_energy=positive_energy=0.
                for group in sorted(set(labels[rows])):
                    ids=np.flatnonzero(labels[rows]==group);entries={str(t):{'created':0,'destroyed':0,'original_positive':0,'decoded_positive':0} for t in spec['thresholds']}
                    ze=pe=0.
                    for start in range(0,len(mapped),256):
                        cols=mapped[start:start+256];a=donors[np.ix_(ids,cols)];b=pred[np.ix_(ids,cols)];delta=np.asarray(b,float)-a
                        ze+=float(np.sum(delta[a==0]**2));pe+=float(np.sum(delta[a>0]**2))
                        for t in spec['thresholds']:
                            original=a>t;decoded=b>t;v=entries[str(t)];v['created']+=int((~original&decoded).sum());v['destroyed']+=int((original&~decoded).sum());v['original_positive']+=int(original.sum());v['decoded_positive']+=int(decoded.sum())
                    counts.append({'lineage':group,'cells':len(ids),'entries':len(ids)*len(mapped),'threshold_counts':entries,'zero_origin_displacement_squared':ze,'positive_origin_displacement_squared':pe});zero_energy+=ze;positive_energy+=pe
                v=variograms(donors,pred,pairs);delta=v['decoded']-v['donor'];z=v['zero_origin']-v['donor'];p=v['positive_origin']-v['donor'];interaction=delta-z-p
                result={'cutoff':cutoff,'candidate':name,'prediction_sha256':sha,'exact_replay_passed':True,'max_fit_stage':guard.max_stage,'lineages':counts,'zero_origin_displacement_squared':zero_energy,'positive_origin_displacement_squared':positive_energy,
                        'selected_feature_covariance_change':covariance_change(donors[:,cov_features],pred[:,cov_features]) if name!='copy' else 0.,
                        'donor_variogram_displacement_mse':float(np.mean(delta**2)),'zero_origin_variogram_mse':float(np.mean(z**2)),'positive_origin_variogram_mse':float(np.mean(p**2)),'interaction_variogram_mse':float(np.mean(interaction**2)),
                        'headline_score':None,'raw_metrics':None,'skills':None,'reward_delta':0}
                results.append(result);append_event(events,'support_decomposed',cutoff=cutoff,candidate=name,prediction_sha256=sha,donor_variogram_displacement_mse=result['donor_variogram_displacement_mse']);del pred
        del program,net,donors
    report={'status':'completed','plan':plan,'results':results,'pairs_sha256':digest(out/'pairs.npz'),'target_expression_read':False,'full_panel_scoring_executed':False,'reward_delta':0}
    (out/'report.json').write_text(json.dumps(report,indent=2));(HERE/PUBLIC).write_text(json.dumps({**report,'report_sha256':digest(out/'report.json')},indent=2));append_event(events,'audit_completed',report_sha256=digest(out/'report.json'))


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
