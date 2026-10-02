"""Matched first paper-objective pilot. No official upload or hidden future reads."""
import copy
import json
import traceback
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from cnf_manifold_flow import DensityFlowNet, train_manifold_density
from cnf_covariance_alignment import CovarianceAlignedTrajectory
from full_anchor_slope_forecast import FullAnchorSlopeForecast
from hurdle_backtest import read_cells
from temporary_forecast_cache import TemporaryForecastCache
from offline_backtest import load_core, Panel
from run_t1 import digest
from iterate import now, append_event
from scnode_biological import train_pair, AnchoredNeuralForecast

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = HERE / 'private/scnode_biological_pilot_01'
PUBLIC = HERE / 'SCNODE_BIOLOGICAL_PILOT_RESULTS.json'


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    if RUN.exists() or PUBLIC.exists():
        raise ValueError('Never duplicate paper pilot')
    RUN.mkdir(parents=True)
    events = RUN / 'events.jsonl'
    emit = lambda kind, **kw: append_event(events, kind, **kw)
    cache = TemporaryForecastCache(HERE / 'private/temporary_cache')
    report = {'status': 'running', 'panels': [], 'new_scoring_batch': True}
    try:
        spec_path = HERE / 'NEXT_SCNODE_BIOLOGICAL_PILOT.json'
        spec = json.loads(spec_path.read_text())
        old = HERE / 'private/cnf_covariance_alignment_01'
        archive = HERE / 'private/cnf_feature_challenge_01'
        prepared = HERE / 'private/associated_prepared_01'
        historical = json.loads((old / 'report.json').read_text())
        original_plan = json.loads((old / 'plan.json').read_text())
        frozen = json.loads((old / 'generation.json').read_text())
        names = spec['candidates']
        sources = ['scnode_biological_pilot.py','scnode_biological.py','scnode_joint_model.py','cnf_manifold_flow.py','cnf_density_flow.py',
                   'full_anchor_slope_forecast.py','log1p_positive_forecast.py','cnf_covariance_alignment.py']
        plan = {'created_utc':now(),'spec_sha256':digest(spec_path),'source_sha256':{f:digest(HERE/f) for f in sources},
                'initial_checkpoint_sha256':digest(archive/'features4096.pt'), 'encoder_sha256':digest(archive/'encoder4096.npz'),
                'historical_plan_sha256':digest(old/'plan.json'),'candidate_names':names,'cutoff':8.5,'target':9.5,
                'seed':spec['seed'],'scoring_seeds':[20260928,20260929,20260930],'fit_scope':'past only <=8.5',
                'steps':1000,'pretraining_steps':200,'batch_size':32,'joint_training_seed':20261002,'submissions_allowed':0}
        save(RUN/'plan.json',plan)
        report['plan_sha256'] = digest(RUN/'plan.json')
        emit('plan_frozen',sha256=report['plan_sha256'])
        for f,h in original_plan['source_sha256'].items():
            if digest(HERE/f)!=h:
                raise ValueError('Historical control source changed: '+f)
        if plan['initial_checkpoint_sha256'] != original_plan['flow_sha256'] or plan['encoder_sha256'] != original_plan['encoder_sha256']:
            raise ValueError('Original checkpoint mismatch')
        for path,h in original_plan['input_sha256'].items():
            if digest(ROOT/path)!=h:
                raise ValueError('Historical input changed')
        prep = json.loads((prepared/'report.json').read_text())
        for f,k in [('expression.npy','expression_sha256'),('genes.csv','genes_sha256'),('selected_metadata.csv','metadata_sha256')]:
            if digest(prepared/f)!=prep[k]:
                raise ValueError('Prepared input changed')
        panel = (ROOT/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
        donors, rows = read_cells(ROOT/'data/E8.5_RNA.h5ad',panel,1500,20260928)
        np.testing.assert_array_equal(rows,np.load(old/'donor_rows.npy'))
        x = np.load(prepared/'expression.npy',mmap_mode='r')
        stages = pd.read_csv(prepared/'selected_metadata.csv').numeric_stage.to_numpy(float)
        symbols = pd.read_csv(prepared/'genes.csv').symbol.fillna('').tolist()
        with np.load(archive/'encoder4096.npz') as e:
            features,center,scale = e['features'],e['center'],e['scale']
            original = DensityFlowNet(e['basis'],e['pca_center'],8.5,7.25)
        original.load_state_dict(torch.load(archive/'features4096.pt',weights_only=False,map_location='cpu')['net'])
        original.eval()
        program = FullAnchorSlopeForecast(x,stages,8.5,donors,panel,symbols,original,center,scale,features,
                                         np.load(archive/'features.npy'),anchor_path=ROOT/'data/E8.5_RNA.h5ad')
        program.configure(.25,.75)
        with np.load(old/'alignment.npz') as a:
            source_mean,challenge_mean = a['source_mean'],a['challenge_mean']
            transform = np.eye(len(source_mean))+.25*(a['full_map']-np.eye(len(source_mean)))
        counts = Counter(symbols)
        lookup = {g:i for i,g in enumerate(symbols) if g and counts[g]==1}
        mapped=np.array([i for i,g in enumerate(panel) if g in lookup],dtype=int)
        columns=np.array([lookup[panel[i]] for i in mapped],dtype=int)
        np.save(RUN/'mapped.npy',mapped);np.save(RUN/'columns.npy',columns)
        emit('past_training_scope_frozen',fit_max_stage=float(stages[stages<=8.5].max()),mapped_genes=len(mapped),protected_genes=len(panel)-len(mapped))
        flows=train_pair(x,stages,columns,8.5,RUN,emit)
        generation = {}
        for name in names:
            if name=='copy':
                pred,ids,audit = donors.copy(),np.arange(len(donors)),{'method':'persistence'}
            elif name=='incumbent_covariance_0.25':
                program.net = CovarianceAlignedTrajectory(original,source_mean,challenge_mean,transform)
                pred,ids,audit = program.predict(9.5,'joint',1.,sampling='systematic')
            else:
                new_program=AnchoredNeuralForecast(x,stages,8.5,donors,mapped,columns,flows[name],np.load(archive/'features.npy'),ROOT/'data/E8.5_RNA.h5ad')
                pred,ids,audit=new_program.predict(9.5,'joint',1.,sampling='systematic')
                protected=np.setdiff1d(np.arange(len(panel)),mapped)
                np.testing.assert_array_equal(pred[:,protected],donors[:,protected])
                del new_program
            sha = cache.put(name,pred)
            expected_name = 'copy' if name=='copy' else 'covariance_0.25'
            if name in ['copy','incumbent_covariance_0.25']:
                if sha!=frozen[expected_name]['prediction_sha256']:
                    raise ValueError('Exact incumbent control replay failed')
                np.testing.assert_array_equal(ids,np.load(old/(expected_name+'_indices.npy')))
            np.save(RUN/(name+'_indices.npy'),ids)
            generation[name]={'prediction_sha256':sha,'audit':audit}
            emit('forecast_frozen',candidate=name,prediction_sha256=sha)
            del pred
        report['generation'] = generation
        emit('all_forecasts_frozen_before_target_expression')
        core,_ = load_core()
        for seed in plan['scoring_seeds']:
            future,target_rows = read_cells(ROOT/'data/E9.5_RNA.h5ad',panel,2000,seed)
            np.testing.assert_array_equal(target_rows,np.load(old/f'target_rows_{seed}.npy'))
            order = np.random.default_rng(seed).permutation(len(future))
            evaluator = Panel(core,future[order[:1000]],donors,seed)
            floor,ceiling = evaluator.metrics(donors),evaluator.metrics(future[order[1000:]])
            prior = next(p for p in historical['panels'] if p['seed']==seed)
            if floor!=prior['floor'] or ceiling!=prior['ceiling']:
                raise ValueError('Historical scorer/calibration changed')
            outcomes=[]
            for name in names:
                with cache.read(name,consume=False) as pred:
                    raw=evaluator.metrics(pred)
                row={'candidate':name,'prediction_sha256':generation[name]['prediction_sha256'],
                     'raw_metrics':raw,**evaluator.aggregate(raw,floor,ceiling)}
                outcomes.append(row)
                emit('candidate_scored',seed=seed,**row)
            report['panels'].append({'seed':seed,'cutoff':8.5,'target':9.5,'floor':floor,'ceiling':ceiling,'results':outcomes})
            save(RUN/'report.partial.json',report)
        summary=[]
        for name in names:
            rows=[next(r for r in p['results'] if r['candidate']==name) for p in report['panels']]
            summary.append({'candidate':name,'scores':[r['local_score'] for r in rows],
                            'mean_score':float(np.mean([r['local_score'] for r in rows])),
                            'raw_metrics':[r['raw_metrics'] for r in rows], 'skills':[r['skills'] for r in rows],
                            'all_calibrations_valid':all(r['calibration_valid'] for r in rows),
                            'mean_skills':{m:float(np.mean([r['skills'][m] for r in rows])) for m in rows[0]['skills']}})
        by={r['candidate']:r for r in summary}
        new=by['scnode_beta_0.1']
        controls=[by['incumbent_covariance_0.25'],by['scnode_beta_0.0']]
        passed=new['all_calibrations_valid'] and all(c['all_calibrations_valid'] for c in controls) and all(
            new['scores'][i]>max(c['scores'][i] for c in controls) for i in range(3)) and all(
            new['mean_skills'][m]>=max(c['mean_skills'][m] for c in controls) for m in new['mean_skills'])
        report.update(status='completed',completed_utc=now(),summary=summary,folds=report['panels'],
                      passing_candidates=['scnode_beta_0.1'] if passed else [],
                      expand_to_16_resamples=bool(passed and new['mean_score']>=60),
                      original_readiness_gate_passed=False,official_score=None,
                      scope='First fullmappedgene joint VAE/ODE dynamic regularization adaptation, not faithful scNODE; exposed one-day development, no independent embryos or official forecast score.')
    except Exception as exc:
        (RUN/'traceback.txt').write_text(traceback.format_exc())
        report.update(status='failed',error=type(exc).__name__+': '+str(exc),failed_utc=now(),passing_candidates=[])
    finally:
        cache.close()
    save(RUN/'report.json',report)
    report['report_sha256']=digest(RUN/'report.json')
    save(PUBLIC,report)
    emit('batch_finished',status=report['status'],report_sha256=report['report_sha256'])
    if report.get('panels'):
        from index_scores import main as index_scores
        index_scores()
    print(json.dumps({'status':report['status'],'report':str(PUBLIC)}))


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        main()
