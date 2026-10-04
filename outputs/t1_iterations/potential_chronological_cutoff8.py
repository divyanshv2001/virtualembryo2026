"""New strict cutoff8 potential coverage; cached lower-cutoff CNF, no future fitting."""
import json
import sys
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now,append_event
from potential_real_anchor_transfer import drift
from scalar_potential_residual import PotentialResidualFlow

RUN=HERE/'private/potential_chronological_cutoff8_01'
PUBLIC=HERE/'POTENTIAL_CHRONOLOGICAL_CUTOFF8_RESULTS.json'


def main():
    from collections import Counter
    from scnode_past_fold_training import PastOnlyMatrix
    from tigon_conditional_fullpanel import panel_values
    from full_anchor_slope_forecast import FullAnchorSlopeForecast
    from cnf_covariance_alignment import CovarianceAlignedTrajectory
    from temporary_forecast_cache import TemporaryForecastCache
    from frozen_fullpanel_scoring import score_frozen_forecasts
    from offline_backtest import load_core
    if RUN.exists() or PUBLIC.exists():raise ValueError('Never duplicate evidence')
    old=HERE/'private/scnode_past_fold_retry_01'
    data=HERE/'private/associated_prepared_01'
    state=json.loads((HERE/'LOCAL_OPTIMIZATION_STATE.json').read_text())
    for filename,sha in state['chronological_crossfit_feasibility']['cache_sha256'].items():
        if digest(HERE/filename)!=sha:raise ValueError('Lower-cutoff cache changed')
    prepared=json.loads((data/'report.json').read_text())
    for name,key in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(data/name)!=prepared[key]:raise ValueError('Prepared source changed')
    panel=(HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
    enc=dict(np.load(old/'fresh_encoder.npz'))
    stages=pd.read_csv(data/'selected_metadata.csv').numeric_stage.to_numpy(float)
    symbols=pd.read_csv(data/'genes.csv').symbol.fillna('').tolist()
    if len(panel)!=32285 or stages[enc['pca_rows']].max()>8.:raise ValueError('Panel/cutoff mismatch')
    raw_x=np.load(data/'expression.npy',mmap_mode='r')
    x=PastOnlyMatrix(raw_x,stages,8.)
    past=np.flatnonzero(stages<=8.)
    if len(past)!=7963:raise ValueError('Source scope changed')
    sources=[Path(__file__),HERE/'potential_real_anchor_transfer.py',HERE/'scalar_potential_residual.py',
             HERE/'cnf_manifold_flow.py',HERE/'full_anchor_slope_forecast.py',HERE/'cnf_covariance_alignment.py',
             HERE/'frozen_fullpanel_scoring.py',HERE/'offline_backtest.py',HERE/'scnode_past_fold_training.py']
    plan={'created_utc':now(),'cutoff':8.,'targets':[8.25,8.5],'scoring_seeds':[20260928,20260929,20260930],
          'fit_stages':[7.5,7.75,8.],'fit_rows':7963,'steps':400,'training_seed':20261004,'batch_size':64,
          'candidates':['copy','cnf800','potential400'],'primary_contrast':['cnf800','potential400'],
          'source_sha256':{str(p):digest(p) for p in sources},'cached_sha256':state['chronological_crossfit_feasibility']['cache_sha256'],
          'prepared_report_sha256':digest(data/'report.json'),'heads':[.25,.75],
          'hypothesis':'Does previously tested potential residual generalize to a new strict earlier training cutoff and two horizons?',
          'scope':'Source-only chronological mechanism coverage; targets historically exposed, one training seed/three resamples, not independent embryos or true out-of-fold selection.',
          'fixed_alignment':'Archived near-identity .25 covariance wrapper, unchanged; no new alignment grid.',
          'official_score':None,'readiness_gates_unchanged':True,'submissions_allowed':0}
    RUN.mkdir();(RUN/'plan.json').write_text(json.dumps(plan,indent=2))
    emit=lambda event,**kw:append_event(RUN/'events.jsonl',event,**kw)
    emit('plan_frozen',sha256=digest(RUN/'plan.json'))
    counts=Counter(symbols);lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
    mapped=np.array([i for i,s in enumerate(panel) if s in lookup]);columns=np.array([lookup[panel[i]] for i in mapped])
    features=np.array([lookup[panel[i]] for i in enc['features']])
    raw=np.asarray(x[np.ix_(past,features)],np.float32)
    coordinates=((raw-enc['center'])/enc['scale']-enc['pca_center'])@enc['basis'].T
    np.savez_compressed(RUN/'context.npz',coordinates=coordinates,stages=stages[past]);del raw,coordinates
    packet={'context':str(RUN/'context.npz'),'encoder':str(old/'fresh_encoder.npz'),'drift':str(old/'fresh_reference.pt'),
            'cutoff':8.,'fit_stages':[7.5,7.75,8.],'fit_rows':7963}
    packet['sha256']={str(p):digest(p) for p in sources+[Path(packet[k]) for k in ['context','encoder','drift']]}
    (RUN/'training_packet.json').write_text(json.dumps(packet,indent=2))
    subprocess.run([str(HERE.parents[1]/'outputs/research_workflow/.venv_cuda/Scripts/python.exe'),str(Path(__file__)),
                    '--train-packet',str(RUN/'training_packet.json')],check=True)
    initial=drift(enc,old/'fresh_reference.pt',8.)
    potential=PotentialResidualFlow(initial,'potential')
    potential.load_state_dict(torch.load(RUN/'potential400.pt',map_location='cpu',weights_only=False)['net']);potential.eval()
    donor_rows=np.load(old/'donor_rows.npy')
    if not np.all(stages[donor_rows]==8.):raise ValueError('Future donors')
    donors=panel_values(x,donor_rows,mapped,columns,len(panel))
    alignment=dict(np.load(old/'fresh_alignment.npz'))
    cache=TemporaryForecastCache(HERE/'private/temporary_cache')
    generation={}
    try:
        for candidate,field in [('copy',None),('cnf800',initial),('potential400',potential)]:
            if field is not None:
                program=FullAnchorSlopeForecast(x,stages,8.,donors,panel,symbols,field,enc['center'],enc['scale'],
                    enc['features'],enc['guard_features'],anchor_path=old/'source_E8.0_anchor.h5ad')
                program.configure(.25,.75)
                program.net=CovarianceAlignedTrajectory(field,alignment['source_mean'],alignment['challenge_mean'],alignment['transform'])
            for target in plan['targets']:
                key=f'{candidate}_{target}'
                if field is None:pred=donors.copy();audit={'method':'source8.0 persistence'}
                else:
                    if candidate=='potential400':
                        field.residual_enabled=False
                        replay,_,_=program.predict(target,'joint',1.,sampling='systematic')
                        replay_key='disabled_'+str(target)
                        if cache.put(replay_key,replay)!=generation[f'cnf800_{target}']['prediction_sha256']:raise ValueError('Disabled exact replay failed')
                        with cache.read(replay_key):pass
                        del replay;field.residual_enabled=True
                    pred,_,audit=program.predict(target,'joint',1.,sampling='systematic')
                generation[key]={'prediction_sha256':cache.put(key,pred),'audit':audit};del pred
                emit('forecast_frozen',candidate=key,sha256=generation[key]['prediction_sha256'])
            if field is not None:del program
        reference=json.loads((old/'report.json').read_text())
        expected=reference['generation']['fresh_pca8_cnf_incumbent_covariance_0.25']['prediction_sha256']
        if generation['cnf800_8.5']['prediction_sha256']!=expected:raise ValueError('Archived8.5 forecast exact replay failed')
        emit('all_forecasts_frozen_before_target_scoring')
        core,_=load_core();folds=[]
        for target in plan['targets']:
            folder=RUN/f'horizon_{target}';folder.mkdir()
            (folder/'plan.json').write_text(json.dumps(plan,indent=2))
            names=[f'{c}_{target}' for c in plan['candidates']]
            report={'plan':plan,'panels':[],'generation':{n:generation[n] for n in names}}
            score_frozen_forecasts(core,raw_x,stages,donors,mapped,columns,panel,8.,target,plan,names,report['generation'],cache,folder,report,emit,
                {'control_candidates':names[:2],'contrast_candidates':names[1:],'scope':plan['scope']})
            if target==8.5:
                for a,b in zip(report['panels'],reference['panels']):
                    if a['floor']!=b['floor'] or a['ceiling']!=b['ceiling']:raise ValueError('Archived calibration exact replay failed')
                    actual=next(r for r in a['results'] if r['candidate']=='cnf800_8.5')['raw_metrics']
                    prior=next(r for r in b['results'] if r['candidate']=='fresh_pca8_cnf_incumbent_covariance_0.25')['raw_metrics']
                    if actual!=prior:raise ValueError('Archived scorer exact replay failed')
            folds.append(report)
        peak=max(f['resource_peak_process_working_set_bytes'] for f in folds)
        if peak>16*1024**3:raise ValueError('Host memory cap')
        result={'status':'completed','plan':plan,
                'horizon_summaries':[{'target':f['panels'][0]['target'],'contrast':f['growth_enabled_minus_disabled_mean_skills']} for f in folds],
                'resource_peak_process_working_set_bytes':peak,'panels':[p for f in folds for p in f['panels']],
                'summary':[s for f in folds for s in f['summary']],'scope':plan['scope'],
                'passing_candidates':[n for f in folds for n in f['passing_candidates']],
                'checkpoint_sha256':digest(RUN/'potential400.pt'),'local_72_gate_passed':False,'official_score':None}
        (RUN/'report.json').write_text(json.dumps(result,indent=2));result['report_sha256']=digest(RUN/'report.json')
        PUBLIC.write_text(json.dumps(result,indent=2));emit('completed',report_sha256=result['report_sha256'])
    finally:cache.close()


if __name__=='__main__':
    if '--train-packet' in sys.argv:
        import potential_real_anchor_transfer as training
        training.RUN=RUN;training.DIRECT=True;training.MMD=False
        training.train(sys.argv[sys.argv.index('--train-packet')+1])
    else:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(2):
            torch.set_num_threads(2)
            try:main()
            except Exception as error:
                if RUN.exists() and not PUBLIC.exists():
                    r={'status':'failed','error':str(error),'scope':'Strictcutoff8 chronological coverage; failure notvalidatedscore.'}
                    (RUN/'report.json').write_text(json.dumps(r,indent=2));r['report_sha256']=digest(RUN/'report.json');PUBLIC.write_text(json.dumps(r,indent=2))
                raise
