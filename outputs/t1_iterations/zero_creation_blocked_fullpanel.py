"""Zero-creation-blocked decoder ablation; exact additive control replay, fixed full-panel scorer."""
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
from forecast_transform_fidelity import implied_mass
from temporary_forecast_cache import TemporaryForecastCache
from cnf_manifold_flow import DensityFlowNet
from partial_anchor_forecast import PartialAnchorForecast
from offline_backtest import load_core,Panel
from metric_critique_reward import assess
from zero_creation_blocked_decoder import decode_blocked,numerical_controls
from robust_population import covariance_change

RUN='zero_creation_blocked_fullpanel_01'
PUBLIC='ZERO_CREATION_BLOCKED_FULLPANEL_RESULTS.json'


def decode(donors,mapped,donor_labels,saved,supported,name):
    pred=donors.copy()
    for group in supported:
        shift=saved[group+'__'+name]-saved[group+'__persistence'];rows=np.flatnonzero(donor_labels==group)
        for start in range(0,len(mapped),256):
            cols=mapped[start:start+256];block=np.asarray(pred[np.ix_(rows,cols)],float)+shift[start:start+len(cols)]
            pred[np.ix_(rows,cols)]=np.maximum(block,0).astype(np.float32)
    changed_rows=np.flatnonzero(np.isin(donor_labels,supported));mass=implied_mass(donors,mapped);changed_mass=implied_mass(pred,mapped)
    factors=np.divide(mass,changed_mass,out=np.ones_like(mass),where=changed_mass>0)
    for start in range(0,len(mapped),256):
        cols=mapped[start:start+256];a=np.expm1(np.asarray(pred[np.ix_(changed_rows,cols)],float))
        pred[np.ix_(changed_rows,cols)]=np.log1p(a*factors[changed_rows,None]).astype(np.float32)
    return pred


def main():
    out=HERE/'private'/RUN
    if out.exists():raise ValueError('Frozen run exists; never duplicate')
    spec_path=HERE/'NEXT_ZERO_CREATION_BLOCKED_FULLPANEL.json';spec=json.loads(spec_path.read_text())
    source=HERE/'private/associated_prepared_01';prepared=json.loads((source/'report.json').read_text())
    for name,key in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(source/name)!=prepared[key]:raise ValueError('Source changed')
    preflight_path=HERE/'private/capture_mean_decoder_preflight_01/report.json';preflight=json.loads(preflight_path.read_text())
    control_path=HERE/'private/capture_mean_decoder_fullpanel_01/report.json';controls=json.loads(control_path.read_text())
    mean_root=HERE/'private/capture_mean_shrinkage_audit_01';means=json.loads((mean_root/'report.json').read_text())
    metadata=pd.read_csv(source/'selected_metadata.csv');stages=metadata.numeric_stage.to_numpy(float);labels=metadata.celltype_extended_atlas.map(lineage).to_numpy()
    symbols=pd.read_csv(source/'genes.csv').symbol.fillna('').tolist();freq=Counter(symbols)
    panel_path=HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt';panel=panel_path.read_text().splitlines()
    if len(set(panel))!=len(panel):raise ValueError('Duplicate panel genes')
    lookup={s:i for i,s in enumerate(panel)};columns=np.array([i for i,s in enumerate(symbols) if s in lookup and freq[s]==1]);mapped=np.array([lookup[symbols[i]] for i in columns])
    x=np.load(source/'expression.npy',mmap_mode='r');core,scorer_manifest=load_core()
    plan={**spec,'created_utc':now(),'spec_sha256':digest(spec_path),'prepared_report_sha256':digest(source/'report.json'),'panel_sha256':digest(panel_path),'preflight_report_sha256':digest(preflight_path),
          'mean_report_sha256':digest(mean_root/'report.json'),'scorer_manifest':scorer_manifest,'control_report_sha256':digest(control_path),
          'code_sha256':{n:digest(HERE/n) for n in ['zero_creation_blocked_fullpanel.py','capture_mean_decoder_fullpanel.py','offline_backtest.py','partial_anchor_forecast.py','log1p_positive_forecast.py','feature_panel_forecast.py','forecast_transform_fidelity.py','temporary_forecast_cache.py','projection_survival_diagnostics.py','metric_critique_reward.py','zero_creation_blocked_decoder.py','robust_population.py']}}
    plan['archive_sha256']={str(c):{n:digest(HERE/f'private/cnf_hurdle_temporal_01/cutoff_{c}'/n) for n in ['encoder.npz','training.pt','heads.npz']} for c,t in spec['folds']}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2));(out/'executed_source.py').write_bytes((HERE/'zero_creation_blocked_fullpanel.py').read_bytes())
    events=out/'events.jsonl';append_event(events,'plan_frozen',sha256=digest(out/'plan.json'));numerical_controls();append_event(events,'numerical_controls_passed')
    report={'status':'running','plan':plan,'folds':[]};cache=TemporaryForecastCache(HERE/'private/temporary_cache');pending=[]
    def values(rows,reader):
        result=np.zeros((len(rows),len(panel)),np.float32)
        for start in range(0,len(columns),256):result[:,mapped[start:start+256]]=np.asarray(reader[np.ix_(rows,columns[start:start+256])],np.float32)
        return result
    try:
        # Freeze both folds before any future expression enters the evaluator.
        for cutoff,target in spec['folds']:
            folder=out/f'cutoff_{cutoff}';folder.mkdir();guard=PastReadGuard(x,stages,cutoff)
            donor_path=HERE/f'private/source_slope_fullpanel_02/cutoff_{cutoff}/donor_rows.npy';donor_rows=np.load(donor_path)
            if len(donor_rows)!=1500 or np.any(stages[donor_rows]!=cutoff):raise ValueError('Donor split changed')
            np.save(folder/'donor_rows.npy',donor_rows);donors=values(donor_rows,guard)
            arc=HERE/f'private/cnf_hurdle_temporal_01/cutoff_{cutoff}'
            with np.load(arc/'encoder.npz') as enc:
                features=enc['features'];cov_features=enc['guard_features'];center=enc['center'];scale=enc['scale']
                net=DensityFlowNet(enc['basis'],enc['pca_center'],cutoff,7.25)
            net.load_state_dict(torch.load(arc/'training.pt',weights_only=False,map_location='cpu')['net']);net.eval()
            program=PartialAnchorForecast(guard,stages,cutoff,donors,panel,symbols,net,center,scale,features,cov_features);program.configure(.5,.5)
            case=next(c for c in means['fit_cases'] if c['cutoff']==cutoff);mean_path=mean_root/f'cutoff_{cutoff}'/'means.npz'
            if digest(mean_path)!=case['forecast_sha256']:raise ValueError('Mean forecast changed')
            supported=[r['lineage'] for r in case['records'] if r['status']=='supported'];generation={}
            with np.load(mean_path) as saved:
                if not np.array_equal(saved['columns'],columns):raise ValueError('Mapping changed')
                for name in spec['candidates']:
                    if name=='copy':pred=donors.copy();audit={'method':'persistence'}
                    elif name=='anchor_unshrunk':pred,ids,audit=program.predict(target,'joint',1.,sampling='systematic')
                    elif name.startswith('blocked_'):
                        base=name.replace('blocked_','capture_');pred=decode_blocked(donors,mapped,labels[donor_rows],saved,supported,base)
                        protected=np.setdiff1d(np.arange(len(panel)),mapped);fallback=~np.isin(labels[donor_rows],supported)
                        mass=implied_mass(donors,mapped);mass_error=float(np.max(np.abs(implied_mass(pred,mapped)-mass)/np.maximum(mass,1e-9)))
                        mean_change=float(np.max(np.abs(pred.mean(0,dtype=float)-donors.mean(0,dtype=float))));cov_change=covariance_change(donors[:,cov_features],pred[:,cov_features])
                        created=sum(int(((donors[:,mapped[j:j+256]]==0)&(pred[:,mapped[j:j+256]]>0)).sum()) for j in range(0,len(mapped),256))
                        destroyed=sum(int(((donors[:,mapped[j:j+256]]>0)&(pred[:,mapped[j:j+256]]==0)).sum()) for j in range(0,len(mapped),256))
                        fidelity=[]
                        for group in supported:
                            rows=np.flatnonzero(labels[donor_rows]==group)
                            if not len(rows):continue
                            intended=saved[group+'__'+base]-saved[group+'__persistence'];actual=pred[rows][:,mapped].mean(0,dtype=float)-donors[rows][:,mapped].mean(0,dtype=float);norm=float(np.linalg.norm(intended))
                            fidelity.append({'lineage':group,'intended_norm':norm,'decoded_norm':float(np.linalg.norm(actual)),'relative_shift_error':float(np.linalg.norm(actual-intended)/norm) if norm else None})
                        audit={'method':'zero_creation_blocked_logshift_countmass','count_mass_relative_error':mass_error,'max_mean_change':mean_change,'covariance_change':cov_change,'created_positive_entries':created,'destroyed_positive_entries':destroyed,'lineage_shift_fidelity':fidelity}
                        if created or mass_error>1e-5 or mean_change>.5 or cov_change>.4 or not np.array_equal(pred[:,protected],donors[:,protected]) or not np.array_equal(pred[fallback],donors[fallback]):raise ValueError('Blocked decoder guard failure before target read')
                    else:pred=decode(donors,mapped,labels[donor_rows],saved,supported,name);audit={'method':'frozen_capture_mean_countmass_decoder'}
                    if not np.isfinite(pred).all() or np.any(pred<0):raise ValueError('Invalid prediction')
                    key=f'{cutoff}/{name}';sha=cache.put(key,pred)
                    if not name.startswith('blocked_'):
                        control_fold=next(f for f in controls['folds'] if f['cutoff']==cutoff)
                        control_row=next(r for r in control_fold['results'] if r['candidate']==name)
                        if sha!=control_row['prediction_sha256']:raise ValueError('Scored control forecast replay failed')
                        audit['exact_scored_control_replay']=True
                    if name not in ['anchor_unshrunk','blocked_quarter','blocked_shrunk']:
                        expected=next(r for r in preflight['results'] if r['cutoff']==cutoff and r['candidate']==('persistence' if name=='copy' else name))
                        if not expected['guards_passed'] or sha!=expected['prediction_sha256'] or digest(donor_path)!=expected['donor_rows_sha256'] or digest(arc/'encoder.npz')!=expected['encoder_sha256']:raise ValueError('Exact guarded preflight replay failed')
                        audit['exact_preflight_replay']=True
                    generation[name]={'prediction_sha256':sha,'audit':audit};append_event(events,'forecast_frozen',cutoff=cutoff,target=target,candidate=name,prediction_sha256=sha,max_fit_stage=guard.max_stage);del pred
            (folder/'generation.json').write_text(json.dumps(generation,indent=2));pending.append((cutoff,target,donors,generation));del program,net
        append_event(events,'all_folds_forecasts_frozen_before_target_read')
        for cutoff,target,donors,generation in pending:
            folder=out/f'cutoff_{cutoff}';target_rows=np.sort(np.random.default_rng(spec['seed']).choice(np.flatnonzero(stages==target),2000,replace=False));np.save(folder/'target_rows.npy',target_rows)
            future=values(target_rows,x);order=np.random.default_rng(spec['seed']).permutation(len(future));evaluator=Panel(core,future[order[:1000]],donors,spec['seed'])
            floor=evaluator.metrics(donors);ceiling=evaluator.metrics(future[order[1000:]]);rows=[]
            for name in spec['candidates']:
                with cache.read(f'{cutoff}/{name}',consume=True) as pred:raw=floor if name=='copy' else evaluator.metrics(pred)
                row={'candidate':name,'prediction_sha256':generation[name]['prediction_sha256'],'raw_metrics':raw,**evaluator.aggregate(raw,floor,ceiling)};rows.append(row);append_event(events,'candidate_scored',cutoff=cutoff,target=target,**row)
            report['folds'].append({'cutoff':cutoff,'target':target,'floor':floor,'ceiling':ceiling,'results':rows,'generation':generation,'donor_rows_sha256':digest(folder/'donor_rows.npy'),'target_rows_sha256':digest(folder/'target_rows.npy')});(out/'report.partial.json').write_text(json.dumps(report,indent=2));del future,evaluator
        assessments=[];passed=[]
        for name in ['blocked_quarter','blocked_shrunk']:
            candidate=[next(r for r in f['results'] if r['candidate']==name) for f in report['folds']];incumbent=[next(r for r in f['results'] if r['candidate']=='anchor_unshrunk') for f in report['folds']];copy=[next(r for r in f['results'] if r['candidate']=='copy') for f in report['folds']]
            eligible=all(r['calibration_valid'] for r in candidate+incumbent+copy)
            assessments.append({'experiment_id':RUN+'/'+name,'assessment':assess([r['skills'] for r in candidate],[r['skills'] for r in incumbent],eligible=eligible),'plan_sha256':digest(out/'plan.json')})
            if eligible and all(c['local_score']>max(a['local_score'],b['local_score']) and all(c['skills'][k]>=a['skills'][k] for k in c['skills']) for c,a,b in zip(candidate,incumbent,copy)):passed.append(name)
        report.update(status='completed',critic_assessments=assessments,passing_candidates=passed,official_score=None,local_72_gate_passed=False,submissions_used=0)
        (out/'report.json').write_text(json.dumps(report,indent=2));(HERE/PUBLIC).write_text(json.dumps({**report,'report_sha256':digest(out/'report.json'),'plan_sha256':digest(out/'plan.json')},indent=2));append_event(events,'fullpanel_completed',passing_candidates=passed)
    except Exception as exc:
        report.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
        (out/'report.json').write_text(json.dumps(report,indent=2));(HERE/PUBLIC).write_text(json.dumps({**report,'report_sha256':digest(out/'report.json')},indent=2));append_event(events,'run_failed',exception_type=type(exc).__name__,message=str(exc));raise
    finally:cache.close()


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
