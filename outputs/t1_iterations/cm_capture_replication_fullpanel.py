"""Past-only capture-excluded model refits and observed CM odds ablations."""
import json
from collections import Counter
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event
from lineage_residual_screen import lineage
from past_encoder_panel import fit_past_encoder
from cnf_manifold_flow import DensityFlowNet, train_manifold_density
from cm_observation_calibration import CMObservationForecast, observation_offset
from temporary_forecast_cache import TemporaryForecastCache
from offline_backtest import load_core, Panel
from metric_critique_reward import assess, total_reward

RUN = 'cm_capture_replication_fullpanel_01'
PUBLIC = 'CM_CAPTURE_REPLICATION_FULLPANEL_RESULTS.json'
NAMES = ['copy', 'anchor_unshrunk', 'cm_025']


def checkpoint(**changes):
    path = HERE/'LOCAL_OPTIMIZATION_STATE.json'
    state = json.loads(path.read_text()); state.update(changes)
    path.write_text(json.dumps(state, indent=2)+'\n')


def main():
    out = HERE/'private'/RUN
    if out.exists():
        raise ValueError('Preserve frozen run; no automatic retry')
    source = HERE/'private/associated_prepared_01'
    prepared = json.loads((source/'report.json').read_text())
    for name, key in [('expression.npy', 'expression_sha256'), ('selected_metadata.csv', 'metadata_sha256'), ('genes.csv', 'genes_sha256')]:
        if digest(source/name) != prepared[key]:
            raise ValueError('Prepared input changed: '+name)
    metadata = pd.read_csv(source/'selected_metadata.csv')
    stages = metadata.numeric_stage.to_numpy(float)
    samples = metadata['sample'].astype(str).to_numpy()
    labels = metadata.celltype_extended_atlas.map(lineage).to_numpy()
    panel_path = HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt'
    panel = panel_path.read_text().splitlines()
    symbols = pd.read_csv(source/'genes.csv').symbol.fillna('').tolist()
    counts = Counter(symbols); lookup = {s:i for i,s in enumerate(symbols) if s and counts[s] == 1}
    mapped = np.array([i for i,s in enumerate(panel) if s in lookup])
    atlas = np.array([lookup[panel[i]] for i in mapped])
    folds = []
    for cutoff, target in [(8.,9.), (8.25,9.25)]:
        current = metadata.loc[stages == cutoff, 'sample'].astype(str).value_counts()
        # Predeclared SECOND-largest current capture; deterministic tie-break, no target access.
        held = sorted(current.index, key=lambda v:(-int(current[v]), v))[1]
        folds.append({'cutoff':cutoff, 'target':target, 'held_capture':held})
    plan = {'created_utc':now(), 'held_capture_rank':2, 'selection_route_sha256':digest(HERE/'JEV_CM_NEXT_ROUTE_20261001_01.json'), 'folds':folds, 'candidates':NAMES, 'seed':20260928,
            'code_sha256':digest(HERE/'cm_capture_replication_fullpanel.py'),
            'source_sha256':{f:digest(HERE/f) for f in ['cm_observation_calibration.py','past_encoder_panel.py','cnf_manifold_flow.py','partial_anchor_forecast.py','log1p_positive_forecast.py','anchor_slope_calibration.py','feature_panel_forecast.py','offline_backtest.py','lineage_residual_screen.py','temporary_forecast_cache.py']},
            'prepared_report_sha256':digest(source/'report.json'), 'panel_sha256':digest(panel_path),
            'jev_route_sha256':digest(HERE/'JEV_CM_ODDS_ROUTE_20261001_01.json'),
            'fit':'Exclude selected capture from ALL encoder/dynamics/source-head fitting. Refit4096gene8D PCA/800update density10 energy.1 flow, conditional heads ridge1. Current heldcapture anchors may calibrate .5/.5 anchor slopes and frozen observation offsets. No future expression used until every forecast is frozen.',
            'formula':'q=(positive_count+.5)/(n+1); offset=logit(q_observed_current)-logit(q_other_current). Add alpha*offset to BOTH p0/p1 log odds for global or CM-only donor cells; calibrated p0 switching denominators. Conditional-positive head fixed across candidates; mapped RAW library mass restored by existing decoder.',
            'support':'At least20 source/observed current cells per calibration scope; otherwise fail explicitly. Coarse supplied CM labels unvalidated.',
            'controls':'Same heldcapture donors, refitted learner, target rows, full32285 scorer/calibration. Copy and no-observation-calibration .5/.5 anchor incumbent. Fixed seed/no choice from future.',
            'scope':'Two reused one-day source temporal folds with current-capture exclusion. Future samples are different captures; unavailable independent embryo pairing. Not same-embryo longitudinal validation or unseen challenge-domain validation.',
            'truth_count':1000, 'ceiling_count':1000, 'submissions_allowed':0,
            'retention':'D-only memmapped source and disposable prediction cache,2threads,one fold at a time. Persist small models, splits,hashes,metrics/events.',
            'decision':'Advance only if candidate beats copy/incumbent headline and preserves each incumbent skill on BOTH folds. >72 mean/lower-tail64replicate AND temporal gate unchanged.'}
    out.mkdir(); (out/'plan.json').write_text(json.dumps(plan,indent=2))
    (out/'executed_source.py').write_bytes((HERE/'cm_capture_replication_fullpanel.py').read_bytes())
    events = out/'events.jsonl'; append_event(events,'plan_frozen',sha256=digest(out/'plan.json'))
    checkpoint(active_jobs=[RUN],local_process_running=True,active_run_path='private/'+RUN,
               cm_capture_replication_job={'status':'running','plan_sha256':digest(out/'plan.json'),'planned_scores':6})
    x = np.load(source/'expression.npy',mmap_mode='r')
    def values(rows):
        result = np.zeros((len(rows),len(panel)),np.float32)
        result[:,mapped] = x[np.ix_(rows,atlas)]
        return result
    core,_ = load_core()
    report = {'plan':plan,'folds':[],'status':'running','official_score':None,'local_gate_passed':False,
              'scorer_manifest_sha256':digest(HERE/'private/scorer_source/manifest.json')}
    for fold in folds:
        cutoff,target,held = fold['cutoff'],fold['target'],fold['held_capture']
        folder = out/f'cutoff_{cutoff}'; folder.mkdir()
        donor_rows = np.flatnonzero((stages == cutoff)&(samples == held))
        current_source = np.flatnonzero((stages == cutoff)&(samples != held))
        donors = values(donor_rows)
        fit_stages = stages.copy(); fit_stages[samples == held] = np.inf
        encoded = fit_past_encoder(x,fit_stages,cutoff,symbols,panel,budget=4096,dimensions=8)
        past = encoded['past_rows']
        if np.any(samples[past] == held) or np.any(stages[past] > cutoff):
            raise ValueError('Past/capture exclusion failed')
        guard = encoded['features'][:384].copy()
        np.save(folder/'donor_rows.npy',donor_rows)
        np.savez_compressed(folder/'encoder.npz',**{k:v for k,v in encoded.items() if k!='coordinates'},guard_features=guard)
        torch.manual_seed(plan['seed'])
        net = DensityFlowNet(encoded['basis'],encoded['pca_center'],cutoff,float(stages[past].min())-.25)
        train_manifold_density(net,encoded['coordinates'],stages[past],.1,folder/'training.pt',
                               lambda kind,**kw:append_event(events,kind,cutoff=cutoff,**kw),steps=800,density_weight=10.)
        program = CMObservationForecast(x,fit_stages,cutoff,donors,panel,symbols,net,
                       encoded['center'],encoded['scale'],encoded['features'],guard)
        program.configure(.5,.5)
        cm_source = current_source[labels[current_source] == 'cardiomyocyte']
        cm_mask = labels[donor_rows] == 'cardiomyocyte'
        offsets = {'global':np.empty(len(mapped)), 'cm':np.empty(len(mapped))}
        for start in range(0,len(mapped),512):
            sl = slice(start,min(start+512,len(mapped)))
            for scope,rows,mask in [('global',current_source,np.ones(len(donors),bool)),('cm',cm_source,cm_mask)]:
                offsets[scope][sl] = observation_offset(x[np.ix_(rows,atlas[sl])],donors[np.ix_(mask,mapped[sl])])
        np.savez_compressed(folder/'observation_offsets.npz',**offsets)
        append_event(events,'current_only_calibration_frozen',cutoff=cutoff,held_capture=held,
                     source_current_cells=len(current_source),source_cm_cells=len(cm_source),observed_cm_cells=int(cm_mask.sum()),
                     past_rows=len(past),offset_sha256=digest(folder/'observation_offsets.npz'))
        generation = {}; cache = TemporaryForecastCache(HERE/'private/temporary_cache')
        try:
            for name in NAMES:
                if name == 'copy':
                    pred,audit = donors.copy(),{'method':'persistence'}
                else:
                    scope = 'none' if name == 'anchor_unshrunk' else name.split('_')[0]
                    alpha = 0. if scope == 'none' else (.25 if name.endswith('025') else .5)
                    program.set_observation_calibration(offsets['global' if scope == 'none' else scope],
                              cm_mask if scope == 'cm' else np.ones(len(donors),bool),alpha,scope)
                    pred,_,audit = program.predict(target,'joint',1.,sampling='systematic')
                    original_mass = np.expm1(donors[:,mapped].astype(float)).sum(1)
                    mass = np.expm1(pred[:,mapped].astype(float)).sum(1)
                    error = float(np.max(np.abs(mass-original_mass)/np.maximum(original_mass,1e-9)))
                    audit = {**audit,'mapped_raw_mass_relative_error':error}
                    if error > 1e-5: raise ValueError('Mapped mass guard failed')
                generation[name] = {'audit':audit,'prediction_sha256':cache.put(name,pred)}
                del pred
            (folder/'generation.json').write_text(json.dumps(generation,indent=2))
            append_event(events,'all_predictions_frozen_before_target_read',cutoff=cutoff,target=target)
            rows = np.sort(np.random.default_rng(plan['seed']).choice(np.flatnonzero(stages == target),2000,replace=False))
            np.save(folder/'target_rows.npy',rows)
            future = values(rows); order = np.random.default_rng(plan['seed']).permutation(2000)
            evaluator = Panel(core,future[order[:1000]],donors,plan['seed'])
            floor = evaluator.metrics(donors); ceiling = evaluator.metrics(future[order[1000:]])
            results = []
            for name in NAMES:
                with cache.read(name) as pred:
                    raw = evaluator.metrics(pred)
                result = {'candidate':name,'raw_metrics':raw,**evaluator.aggregate(raw,floor,ceiling)}
                results.append(result)
                append_event(events,'candidate_scored',cutoff=cutoff,**result)
                (out/'report.partial.json').write_text(json.dumps({**report,'in_progress_fold':{**fold,'floor':floor,'ceiling':ceiling,'results':results,'generation':generation}},indent=2))
            report['folds'].append({**fold,'floor':floor,'ceiling':ceiling,'results':results,'generation':generation,
                            'donor_rows_sha256':digest(folder/'donor_rows.npy'),'target_rows_sha256':digest(folder/'target_rows.npy')})
            (out/'report.partial.json').write_text(json.dumps(report,indent=2))
        finally:
            cache.close()
        del program,net,encoded,donors,future,evaluator
    assessments = []; summary = []
    for name in NAMES:
        candidate = [next(r for r in f['results'] if r['candidate']==name) for f in report['folds']]
        summary.append({'candidate':name,'scores':[r['local_score'] for r in candidate],
                        'raw_metrics':[r['raw_metrics'] for r in candidate],'skills':[r['skills'] for r in candidate],
                        'all_calibrations_valid':all(r['calibration_valid'] for r in candidate)})
        if name not in ['copy','anchor_unshrunk']:
            incumbent = [next(r for r in f['results'] if r['candidate']=='anchor_unshrunk') for f in report['folds']]
            assessments.append({'experiment_id':RUN+'/'+name,'plan_sha256':digest(out/'plan.json'),
                                'assessment':assess([r['skills'] for r in candidate],[r['skills'] for r in incumbent],eligible=all(r['calibration_valid'] for r in candidate+incumbent))})
    controls = {r['candidate']:r for r in summary}
    passed = [r['candidate'] for r in summary if r['candidate'] not in ['copy','anchor_unshrunk'] and r['all_calibrations_valid'] and all(
              r['scores'][i] > max(controls['copy']['scores'][i],controls['anchor_unshrunk']['scores'][i]) and
              all(r['skills'][i][m] >= controls['anchor_unshrunk']['skills'][i][m] for m in ['de_score','de_direction','mmd_u','variogram']) for i in range(2))]
    report.update(status='completed',critic_assessments=assessments)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    public = {'status':'completed','updated_utc':now(),'summary':summary,'folds':report['folds'],
              'passing_candidates':passed,'report_sha256':digest(out/'report.json'),'plan_sha256':digest(out/'plan.json'),
              'official_score':None,'local_72_gate_passed':False,'submissions_used':0}
    (HERE/PUBLIC).write_text(json.dumps(public,indent=2))
    ledger = HERE/'METRIC_CRITIQUE_REWARD_LEDGER.jsonl'
    previous = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    known = {r['experiment_id'] for r in previous}; additions = [r for r in assessments if r['experiment_id'] not in known]
    with ledger.open('a') as f:
        for r in additions:
            r['report_sha256'] = public['report_sha256']; f.write(json.dumps(r)+'\n')
    totals = total_reward(previous+additions)
    state = json.loads((HERE/'LOCAL_OPTIMIZATION_STATE.json').read_text())
    state['metric_critique_reward'].update(current_reward=totals['reward_score'],uncapped_reward=totals['uncapped_reward'])
    checkpoint(active_jobs=[],local_process_running=False,active_run_path=None,metric_critique_reward=state['metric_critique_reward'],
               cm_capture_replication_job={'status':'completed','passing_candidates':passed,'report_sha256':public['report_sha256']})
    append_event(events,'batch_completed',passing_candidates=passed,full_panel_scores=6)
    from index_scores import main as index_scores
    index_scores()


if __name__ == '__main__':
    torch.set_num_threads(2)
    try:
        with threadpool_limits(limits=2): main()
    except Exception as exc:
        out = HERE/'private'/RUN
        if out.exists():
            append_event(out/'events.jsonl','run_failed',exception_type=type(exc).__name__,message=str(exc))
            checkpoint(active_jobs=[],local_process_running=False,active_run_path=None,
                       cm_capture_replication_job={'status':'failed','error_type':type(exc).__name__,'error':str(exc)})
        raise
