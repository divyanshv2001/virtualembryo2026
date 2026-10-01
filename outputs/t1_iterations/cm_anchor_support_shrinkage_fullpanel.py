"""Frozen archived-model replay and strict CM forecast-margin projection."""
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
from cnf_manifold_flow import DensityFlowNet
from cm_observation_calibration import CMObservationForecast, observation_offset
from joint_margin_solver import solve
from robust_population import covariance_change
from temporary_forecast_cache import TemporaryForecastCache
from offline_backtest import load_core, Panel
from metric_critique_reward import assess, total_reward, METRICS
from cm_capture_replication_fullpanel import checkpoint

RUN = 'cm_anchor_support_shrinkage_fullpanel_01'
PUBLIC = 'CM_ANCHOR_SUPPORT_SHRINKAGE_FULLPANEL_RESULTS.json'
NAMES = ['copy', 'anchor_unshrunk', 'cm_025_replay', 'anchor_support_initial_025', 'anchor_projected_identity', 'anchor_support_projected_025']


def initialize(candidate, anchor, cm_mask, mapped):
    result = candidate.copy()
    corrected = candidate[np.ix_(cm_mask,mapped)]
    reference = anchor[np.ix_(cm_mask,mapped)]
    result[np.ix_(cm_mask,mapped)] = np.where(reference>0,np.where(corrected>0,corrected,reference),0.)
    initialized = result[np.ix_(cm_mask,mapped)].astype(float)
    result[np.ix_(cm_mask,mapped)] = (reference.astype(float)+.25*(initialized-reference.astype(float))).astype(np.float32)
    return result


def project(candidate, reference, donors, cm_mask, mapped, guard):
    block = candidate[np.ix_(cm_mask, mapped)]
    target = reference[np.ix_(cm_mask, mapped)].astype(float)
    value, audit = solve(block, target, adaptive=True, iterations=100,
                         mean_tolerance=1e-5, mass_tolerance=1e-5)
    if not audit['valid']:
        return None, audit
    cast = value.astype(np.float32)
    mean_error = float(np.max(np.abs(cast.astype(float).mean(0)-target.mean(0))))
    mass = np.expm1(target).sum(1)
    mass_error = float(np.max(np.abs(np.expm1(cast.astype(float)).sum(1)-mass)/np.maximum(mass,1e-12)))
    result = candidate.copy(); result[np.ix_(cm_mask,mapped)] = cast
    protected = np.ones(candidate.shape[1],bool); protected[mapped] = False
    checks = {'finite_nonnegative':bool(np.isfinite(result).all() and (result>=0).all()),
              'support_preserved':bool(np.array_equal(cast>0,block>0)),
              'protected_cells_unchanged':bool(np.array_equal(result[~cm_mask],candidate[~cm_mask])),
              'protected_genes_unchanged':bool(np.array_equal(result[:,protected],candidate[:,protected])),
              'gene_margin_passed':mean_error<=1e-5,'raw_mass_margin_passed':mass_error<=1e-5}
    covariance = float(covariance_change(donors[:,guard],result[:,guard]))
    checks['covariance_guard_passed'] = bool(np.isfinite(covariance) and covariance<=.4)
    valid = all(checks.values())
    audit.update(valid=valid,post_cast_checks=checks,max_gene_log_mean_error=mean_error,
                 max_row_raw_mass_relative_error=mass_error,covariance_change_vs_reference=covariance,
                 constraints_source='No-offset anchor FORECAST; no future truth',projected_cells=int(cm_mask.sum()))
    if not valid: audit['reason'] = 'Post-float32 margin/support/protection/covariance check failed'
    return (result if valid else None), audit


def main():
    out = HERE/'private'/RUN
    if out.exists(): raise ValueError('Preserve frozen run; no automatic retry')
    spec_path = HERE/'NEXT_ANCHOR_SUPPORT_SHRINKAGE_EXPERIMENT.json'
    spec = json.loads(spec_path.read_text())
    if spec['beta'] != .25: raise ValueError('Undeclared redistribution strength')
    source = HERE/'private/associated_prepared_01'
    prepared = json.loads((source/'report.json').read_text())
    for name,key in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(source/name)!=prepared[key]: raise ValueError('Prepared input changed: '+name)
    archives = {}; artifact_hashes = {}
    for condition in spec['conditions']:
        root = HERE/'private'/condition['archive']; folder = root/f"cutoff_{condition['cutoff']}"
        old = archives.setdefault(condition['archive'],json.loads((root/'report.json').read_text()))
        if old['status']!='completed': raise ValueError('Archive incomplete')
        for name in ['encoder.npz','training.pt','donor_rows.npy','target_rows.npy','observation_offsets.npz','generation.json']:
            artifact_hashes[str((folder/name).relative_to(HERE))] = digest(folder/name)
        artifact_hashes[str((root/'report.json').relative_to(HERE))] = digest(root/'report.json')
        # A changed decoder cannot masquerade as an archived matched replay.
        for name,sha in old['plan']['source_sha256'].items():
            if digest(HERE/name)!=sha: raise ValueError('Archived dependency changed: '+name)
    plan = {**spec,'created_utc':now(),'spec_sha256':digest(spec_path),
            'code_sha256':digest(HERE/'cm_anchor_support_shrinkage_fullpanel.py'),
            'source_sha256':{name:digest(HERE/name) for name in ['joint_margin_solver.py','support_mass_bounds.py','cm_observation_calibration.py','offline_backtest.py']},
            'archive_artifact_sha256':artifact_hashes,'prepared_report_sha256':digest(source/'report.json')}
    out.mkdir(); (out/'plan.json').write_text(json.dumps(plan,indent=2))
    (out/'executed_source.py').write_bytes((HERE/'cm_anchor_support_shrinkage_fullpanel.py').read_bytes())
    events = out/'events.jsonl'; append_event(events,'plan_frozen',sha256=digest(out/'plan.json'))
    checkpoint(active_jobs=[RUN],local_process_running=True,active_run_path='private/'+RUN,
               cm_anchor_support_shrinkage_job={'status':'running','planned_scores':24})
    metadata = pd.read_csv(source/'selected_metadata.csv')
    stages = metadata.numeric_stage.to_numpy(float); samples = metadata['sample'].astype(str).to_numpy()
    labels = metadata.celltype_extended_atlas.map(lineage).to_numpy()
    panel_path = HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt'
    panel = panel_path.read_text().splitlines(); symbols = pd.read_csv(source/'genes.csv').symbol.fillna('').tolist()
    counts = Counter(symbols); lookup = {s:i for i,s in enumerate(symbols) if s and counts[s]==1}
    mapped = np.array([i for i,s in enumerate(panel) if s in lookup]); atlas = np.array([lookup[panel[i]] for i in mapped])
    if len(panel)!=32285: raise ValueError('Full panel changed')
    for old in archives.values():
        if digest(panel_path)!=old['plan']['panel_sha256']: raise ValueError('Archived panel changed')
    x = np.load(source/'expression.npy',mmap_mode='r')
    def values(rows):
        result = np.zeros((len(rows),len(panel)),np.float32); result[:,mapped] = x[np.ix_(rows,atlas)]; return result
    core,_ = load_core()
    report = {'plan':plan,'folds':[],'status':'running','official_score':None,'local_gate_passed':False,
              'scorer_manifest_sha256':digest(HERE/'private/scorer_source/manifest.json')}
    for condition in spec['conditions']:
        cutoff,target,held = condition['cutoff'],condition['target'],condition['held_capture']
        archive = HERE/'private'/condition['archive']/f'cutoff_{cutoff}'
        old = next(f for f in archives[condition['archive']]['folds'] if f['cutoff']==cutoff)
        if old['held_capture']!=held or old['target']!=target: raise ValueError('Archived condition mismatch')
        if report['scorer_manifest_sha256']!=archives[condition['archive']]['scorer_manifest_sha256']: raise ValueError('Scorer changed')
        donor_rows = np.load(archive/'donor_rows.npy'); rows = np.load(archive/'target_rows.npy')
        if digest(archive/'donor_rows.npy')!=old['donor_rows_sha256'] or digest(archive/'target_rows.npy')!=old['target_rows_sha256']: raise ValueError('Archived split changed')
        if not np.all((stages[donor_rows]==cutoff)&(samples[donor_rows]==held)): raise ValueError('Donor mismatch')
        encoded = dict(np.load(archive/'encoder.npz')); past = encoded['past_rows']; guard = encoded['guard_features']
        if np.any(samples[past]==held) or np.any(stages[past]>cutoff): raise ValueError('Past exclusion failed')
        donors = values(donor_rows); fit_stages = stages.copy(); fit_stages[samples==held] = np.inf
        net = DensityFlowNet(encoded['basis'],encoded['pca_center'],cutoff,float(stages[past].min())-.25)
        net.load_state_dict(torch.load(archive/'training.pt',weights_only=False,map_location='cpu')['net']); net.eval()
        program = CMObservationForecast(x,fit_stages,cutoff,donors,panel,symbols,net,encoded['center'],encoded['scale'],encoded['features'],guard)
        program.configure(.5,.5); cm_mask = labels[donor_rows]=='cardiomyocyte'
        offsets = dict(np.load(archive/'observation_offsets.npz'))
        current_cm = np.flatnonzero((stages==cutoff)&(samples!=held)&(labels=='cardiomyocyte'))
        for start in range(0,len(mapped),512):
            sl = slice(start,min(start+512,len(mapped)))
            actual = observation_offset(x[np.ix_(current_cm,atlas[sl])],donors[np.ix_(cm_mask,mapped[sl])])
            if not np.array_equal(actual,offsets['cm'][sl]): raise ValueError('Current-only offset replay mismatch')
        generation = {}; cache = TemporaryForecastCache(HERE/'private/temporary_cache')
        try:
            generation['copy'] = {'audit':{'method':'persistence'},'prediction_sha256':cache.put('copy',donors)}
            for name,alpha,scope in [('anchor_unshrunk',0.,'none'),('cm_025_replay',.25,'cm')]:
                program.set_observation_calibration(offsets['cm' if scope=='cm' else 'global'],cm_mask if scope=='cm' else np.ones(len(donors),bool),alpha,scope)
                pred,_,audit = program.predict(target,'joint',1.,sampling='systematic')
                generation[name] = {'audit':audit,'prediction_sha256':cache.put(name,pred)}
                del pred
            for name in NAMES[:3]:
                archive_name = 'cm_025' if name=='cm_025_replay' else name
                if generation[name]['prediction_sha256']!=old['generation'][archive_name]['prediction_sha256']: raise ValueError('Archived prediction hash mismatch: '+name)
            with cache.read('anchor_unshrunk',consume=False) as anchor, cache.read('cm_025_replay',consume=False) as candidate:
                initial = initialize(candidate,anchor,cm_mask,mapped)
                generation['anchor_support_initial_025'] = {'audit':{'method':'Fixed beta.25 toward anchor after exact anchor support/fill; no future inputs',
                    'covariance_change_vs_reference':float(covariance_change(donors[:,guard],initial[:,guard]))},'prediction_sha256':cache.put('anchor_support_initial_025',initial)}
                identity,identity_audit = project(anchor,anchor,donors,cm_mask,mapped,guard)
                if identity is None: raise ValueError('Projected anchor identity invalid')
                generation['anchor_projected_identity'] = {'audit':identity_audit,'prediction_sha256':cache.put('anchor_projected_identity',identity)}
                if generation['anchor_projected_identity']['prediction_sha256']!=generation['anchor_unshrunk']['prediction_sha256']:
                    raise ValueError('Projected anchor identity hash mismatch')
                del identity
                projected,audit = project(initial,anchor,donors,cm_mask,mapped,guard)
                del initial
            generation['anchor_support_projected_025'] = {'audit':audit,'prediction_sha256':None if projected is None else cache.put('anchor_support_projected_025',projected)}
            del projected
            append_event(events,'all_predictions_frozen_before_target_read',cutoff=cutoff,held_capture=held,generation=generation)
            if not np.all(stages[rows]==target) or len(rows)!=2000: raise ValueError('Target split mismatch')
            future = values(rows); order = np.random.default_rng(spec['seed']).permutation(2000)
            evaluator = Panel(core,future[order[:1000]],donors,spec['seed'])
            floor = evaluator.metrics(donors); ceiling = evaluator.metrics(future[order[1000:]])
            if floor!=old['floor'] or ceiling!=old['ceiling']: raise ValueError('Archived calibration mismatch')
            fold = {**condition,'fold':f'{cutoff}_capture_{held}','floor':floor,'ceiling':ceiling,'results':[],
                    'generation':generation,'donor_rows_sha256':old['donor_rows_sha256'],'target_rows_sha256':old['target_rows_sha256']}
            for name in NAMES:
                if generation[name]['prediction_sha256'] is None:
                    result = {'candidate':name,'local_score':None,'raw_metrics':dict.fromkeys(METRICS),'skills':dict.fromkeys(METRICS),'calibration_valid':False,'invalid_reason':generation[name]['audit'].get('reason','Projection failed')}
                else:
                    with cache.read(name) as pred: raw = evaluator.metrics(pred)
                    result = {'candidate':name,'raw_metrics':raw,**evaluator.aggregate(raw,floor,ceiling)}
                fold['results'].append(result); append_event(events,'candidate_scored_or_invalid',cutoff=cutoff,held_capture=held,**result)
                (out/'report.partial.json').write_text(json.dumps({**report,'in_progress_fold':fold},indent=2))
            report['folds'].append(fold)
            (out/'report.partial.json').write_text(json.dumps(report,indent=2))
        finally: cache.close()
        del program,net,encoded,donors,future,evaluator
    summary = []
    for name in NAMES:
        results = [next(r for r in f['results'] if r['candidate']==name) for f in report['folds']]
        summary.append({'candidate':name,'scores':[r['local_score'] for r in results],'raw_metrics':[r['raw_metrics'] for r in results],
                        'skills':[r['skills'] for r in results],'all_calibrations_valid':all(r['calibration_valid'] for r in results)})
    controls = {r['candidate']:r for r in summary}; anchor = controls['anchor_unshrunk']
    assessments = []; passed = []
    for name in ['anchor_support_initial_025','anchor_support_projected_025']:
        candidate = controls[name]
        assessments.append({'experiment_id':RUN+'/'+name,'plan_sha256':digest(out/'plan.json'),
                  'assessment':assess(candidate['skills'],anchor['skills'],eligible=candidate['all_calibrations_valid'] and anchor['all_calibrations_valid'])})
        if candidate['all_calibrations_valid'] and all(candidate['scores'][i]>max(controls['copy']['scores'][i],anchor['scores'][i],controls['cm_025_replay']['scores'][i]) and all(candidate['skills'][i][m]>=anchor['skills'][i][m] for m in METRICS) for i in range(4)):
            passed.append(name)
    report.update(status='completed',critic_assessments=assessments); (out/'report.json').write_text(json.dumps(report,indent=2))
    public = {'status':'completed','updated_utc':now(),'summary':summary,'folds':report['folds'],'passing_candidates':passed,
              'report_sha256':digest(out/'report.json'),'plan_sha256':digest(out/'plan.json'),'official_score':None,'local_72_gate_passed':False,'submissions_used':0}
    (HERE/PUBLIC).write_text(json.dumps(public,indent=2))
    ledger = HERE/'METRIC_CRITIQUE_REWARD_LEDGER.jsonl'; previous = [json.loads(s) for s in ledger.read_text().splitlines() if s.strip()]
    known = {r['experiment_id'] for r in previous}
    if any(a['experiment_id'] in known for a in assessments): raise ValueError('Duplicate reward')
    with ledger.open('a') as handle:
        for assessment in assessments:
            assessment['report_sha256'] = public['report_sha256']; handle.write(json.dumps(assessment)+'\n')
    totals = total_reward(previous+assessments); state = json.loads((HERE/'LOCAL_OPTIMIZATION_STATE.json').read_text())
    state['metric_critique_reward'].update(current_reward=totals['reward_score'],uncapped_reward=totals['uncapped_reward'])
    checkpoint(active_jobs=[],local_process_running=False,active_run_path=None,metric_critique_reward=state['metric_critique_reward'],
               cm_anchor_support_shrinkage_job={'status':'completed','passing_candidates':public['passing_candidates'],'report_sha256':public['report_sha256']})
    append_event(events,'batch_completed',passing_candidates=public['passing_candidates'],full_panel_scores=sum(r['local_score'] is not None for f in report['folds'] for r in f['results']))
    from index_scores import main as index_scores
    index_scores()


if __name__=='__main__':
    torch.set_num_threads(2)
    try:
        with threadpool_limits(limits=2): main()
    except Exception as exc:
        out = HERE/'private'/RUN
        if out.exists():
            append_event(out/'events.jsonl','run_failed',exception_type=type(exc).__name__,message=str(exc))
            checkpoint(active_jobs=[],local_process_running=False,active_run_path=None,
                       cm_anchor_support_shrinkage_job={'status':'failed','error_type':type(exc).__name__,'error':str(exc)})
        raise
