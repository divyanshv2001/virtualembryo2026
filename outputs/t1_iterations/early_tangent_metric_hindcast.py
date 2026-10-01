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
from metric_critique_reward import assess, total_reward, METRICS
from cm_anchor_support_projection_fullpanel import initialize, project
from early_tangent_controls import construct
from scipy.stats import rankdata

RUN = 'early_tangent_metric_hindcast_01'
PUBLIC = 'EARLY_TANGENT_METRIC_HINDCAST_RESULTS.json'
NAMES = ['copy', 'anchor_unshrunk', 'cm_025', 'support_full_initial', 'support_full_projected', 'support_quarter_initial', 'support_quarter_projected', 'anchor_projected_identity', 'tangent_learned_initial', 'tangent_learned_projected', 'tangent_shuffle_initial', 'tangent_shuffle_projected']


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
    spec = json.loads((HERE/'NEXT_EARLY_TANGENT_METRIC_EXPERIMENT.json').read_text())
    folds = spec['conditions']
    if spec['candidates'] != NAMES: raise ValueError('Declared candidate mismatch')
    plan = {**spec,'created_utc':now(),'folds':folds,
            'predeclaration_sha256':digest(HERE/'NEXT_EARLY_TANGENT_METRIC_EXPERIMENT.json'),
            'code_sha256':digest(HERE/'early_tangent_metric_hindcast.py'),
            'source_sha256':{name:digest(HERE/name) for name in ['early_tangent_controls.py','projection_survival_diagnostics.py','joint_margin_solver.py','cm_anchor_support_projection_fullpanel.py','cm_observation_calibration.py','past_encoder_panel.py','cnf_manifold_flow.py','partial_anchor_forecast.py','log1p_positive_forecast.py','anchor_slope_calibration.py','feature_panel_forecast.py','offline_backtest.py','lineage_residual_screen.py','temporary_forecast_cache.py']},
            'prepared_report_sha256':digest(source/'report.json'),'panel_sha256':digest(panel_path),
            'jev_route_sha256':digest(HERE/'JEV_SURVIVAL_SYNTHESIS_ROUTE_20261001_01.json'),
            'scorer_metric_contract':'DE score uses pseudobulk delta gene ranking; DE direction is reference-partialled rank correlation of pseudobulk deltas. Exact anchor gene means imply same DE components; float32 near-tie sensitivity measured, no eligibility/calibration alteration.'}
    out.mkdir(); (out/'plan.json').write_text(json.dumps(plan,indent=2))
    (out/'executed_source.py').write_bytes((HERE/'early_tangent_metric_hindcast.py').read_bytes())
    events = out/'events.jsonl'; append_event(events,'plan_frozen',sha256=digest(out/'plan.json'))
    checkpoint(active_jobs=[RUN],local_process_running=True,active_run_path='private/'+RUN,
               early_tangent_metric_job={'status':'running','plan_sha256':digest(out/'plan.json'),'planned_scores':24})
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
            generation['copy']={'audit':{'method':'persistence'},'prediction_sha256':cache.put('copy',donors)}
            for name,alpha,scope in [('anchor_unshrunk',0.,'none'),('cm_025',.25,'cm')]:
                program.set_observation_calibration(offsets['cm' if scope=='cm' else 'global'],cm_mask if scope=='cm' else np.ones(len(donors),bool),alpha,scope)
                pred,_,audit=program.predict(target,'joint',1.,sampling='systematic')
                generation[name]={'audit':audit,'prediction_sha256':cache.put(name,pred)}
                del pred
            with cache.read('anchor_unshrunk',consume=False) as anchor,cache.read('cm_025',consume=False) as cm:
                anchor_mean=anchor.mean(0);anchor_mean64=anchor.mean(0,dtype=np.float64)
                full=initialize(cm,anchor,cm_mask,mapped)
                for name,beta in [('support_full',1.),('support_quarter',.25)]:
                    initial=full.copy();initial[np.ix_(cm_mask,mapped)]=(anchor[np.ix_(cm_mask,mapped)].astype(float)+beta*(full[np.ix_(cm_mask,mapped)].astype(float)-anchor[np.ix_(cm_mask,mapped)].astype(float))).astype(np.float32)
                    generation[name+'_initial']={'audit':{'beta':beta},'prediction_sha256':cache.put(name+'_initial',initial)}
                    projected,audit=project(initial,anchor,donors,cm_mask,mapped,guard)
                    generation[name+'_projected']={'audit':audit,'prediction_sha256':None if projected is None else cache.put(name+'_projected',projected)}
                    del initial,projected
                identity,identity_audit=project(anchor,anchor,donors,cm_mask,mapped,guard)
                if identity is None: raise ValueError('Anchor identity invalid')
                generation['anchor_projected_identity']={'audit':identity_audit,'prediction_sha256':cache.put('anchor_projected_identity',identity)}
                if generation['anchor_projected_identity']['prediction_sha256']!=generation['anchor_unshrunk']['prediction_sha256']:raise ValueError('Anchor identity hash mismatch')
                del identity
                try:
                    blocks,construction=construct(anchor[np.ix_(cm_mask,mapped)],full[np.ix_(cm_mask,mapped)],plan['seed'])
                except ValueError as exc:
                    blocks={};construction={'valid':False,'reason':str(exc)}
                for kind in ['learned','shuffle']:
                    if not blocks:
                        for suffix in ['initial','projected']:generation['tangent_'+kind+'_'+suffix]={'audit':construction,'prediction_sha256':None}
                        continue
                    initial=full.copy();initial[np.ix_(cm_mask,mapped)]=blocks[kind]
                    generation['tangent_'+kind+'_initial']={'audit':construction,'prediction_sha256':cache.put('tangent_'+kind+'_initial',initial)}
                    projected,audit=project(initial,anchor,donors,cm_mask,mapped,guard)
                    generation['tangent_'+kind+'_projected']={'audit':{**audit,'construction':construction},'prediction_sha256':None if projected is None else cache.put('tangent_'+kind+'_projected',projected)}
                    del initial,projected
                negative_control={'valid':False,'surviving_norm_ratio':None}
                if generation['tangent_learned_projected']['prediction_sha256'] and generation['tangent_shuffle_projected']['prediction_sha256']:
                    with cache.read('tangent_learned_projected',consume=False) as learned,cache.read('tangent_shuffle_projected',consume=False) as null:
                        ln=float(np.linalg.norm(learned[np.ix_(cm_mask,mapped)].astype(float)-anchor[np.ix_(cm_mask,mapped)].astype(float)))
                        nn=float(np.linalg.norm(null[np.ix_(cm_mask,mapped)].astype(float)-anchor[np.ix_(cm_mask,mapped)].astype(float)))
                    ratio=ln/nn if nn>0 else None
                    negative_control={'valid':bool(ratio is not None and .9<=ratio<=1.1),'surviving_norm_ratio':ratio,'learned_surviving_norm':ln,'shuffle_surviving_norm':nn,'single_shuffle_seed':plan['seed']}
                del full,blocks
            (folder/'generation.json').write_text(json.dumps(generation,indent=2))
            append_event(events,'all_predictions_frozen_before_target_read',cutoff=cutoff,target=target)
            rows = np.sort(np.random.default_rng(plan['seed']).choice(np.flatnonzero(stages == target),2000,replace=False))
            np.save(folder/'target_rows.npy',rows)
            future = values(rows); order = np.random.default_rng(plan['seed']).permutation(2000)
            evaluator = Panel(core,future[order[:1000]],donors,plan['seed'])
            floor = evaluator.metrics(donors); ceiling = evaluator.metrics(future[order[1000:]])
            results = []
            for name in NAMES:
                if generation[name]['prediction_sha256'] is None:
                    result={'candidate':name,'raw_metrics':dict.fromkeys(METRICS),'skills':dict.fromkeys(METRICS),'local_score':None,'calibration_valid':False,'invalid_reason':generation[name]['audit'].get('reason','Invalid projection')}
                else:
                    with cache.read(name) as pred:
                        raw = evaluator.metrics(pred)
                        mean=pred.mean(0);mean64=pred.mean(0,dtype=np.float64);dp=mean-evaluator.ref_mean_source;ap=anchor_mean-evaluator.ref_mean_source
                        order_pred=np.argsort(-np.asarray(dp,float));order_anchor=np.argsort(-np.asarray(ap,float))
                        up=len(evaluator.up);dn=len(evaluator.down)
                        metric_trace={'max_pseudobulk_float32_difference_vsanchor':float(np.max(np.abs(mean-anchor_mean))),
                            'max_pseudobulk_float64_difference_vsanchor':float(np.max(np.abs(mean64-anchor_mean64))),
                            'identical_float32_pseudobulk':bool(np.array_equal(mean,anchor_mean)),
                            'delta_sign_changes_vsanchor':int((np.sign(dp)!=np.sign(ap)).sum()),
                            'delta_gene_rank_changes_vsanchor':int((rankdata(dp)!=rankdata(ap)).sum()),
                            'truth_up_genes':up,'truth_down_genes':dn,
                            'top_up_membership_changes_vsanchor':int(len(np.setxor1d(order_pred[:up],order_anchor[:up]))),
                            'bottom_down_membership_changes_vsanchor':int(len(np.setxor1d(order_pred[-dn:] if dn else [],order_anchor[-dn:] if dn else []))),
                            'signed_overlap':float(core._signed_overlap(dp,evaluator.up,evaluator.down)[0]),
                            'prediction_delta_std':float(np.std(dp)),'truth_delta_std':float(np.std(evaluator.truth_mean-evaluator.ref_mean_source)),
                            'auxiliary_outputs':'PinnedDEgeneeligibility/rankoverlapandpseudobulk available; no inventedgene-wise directioncontribution (correlationnonadditive)'}
                    result = {'candidate':name,'raw_metrics':raw,**evaluator.aggregate(raw,floor,ceiling),'metric_trace':metric_trace}
                results.append(result)
                append_event(events,'candidate_scored',cutoff=cutoff,**result)
                (out/'report.partial.json').write_text(json.dumps({**report,'in_progress_fold':{**fold,'floor':floor,'ceiling':ceiling,'results':results,'generation':generation}},indent=2))
            report['folds'].append({**fold,'floor':floor,'ceiling':ceiling,'results':results,'generation':generation,'negative_control':negative_control,
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
        if name in ['tangent_learned_initial','tangent_learned_projected']:
            incumbent = [next(r for r in f['results'] if r['candidate']=='anchor_unshrunk') for f in report['folds']]
            assessments.append({'experiment_id':RUN+'/'+name,'plan_sha256':digest(out/'plan.json'),
                                'assessment':assess([r['skills'] for r in candidate],[r['skills'] for r in incumbent],eligible=all(r['calibration_valid'] for r in candidate+incumbent))})
    controls = {r['candidate']:r for r in summary}
    passed = [r['candidate'] for r in summary if r['candidate'] in ['tangent_learned_initial','tangent_learned_projected'] and r['all_calibrations_valid'] and controls['tangent_shuffle_projected']['all_calibrations_valid'] and all(f['negative_control']['valid'] for f in report['folds']) and all(
              r['scores'][i] > max(controls['copy']['scores'][i],controls['anchor_unshrunk']['scores'][i],controls['cm_025']['scores'][i],controls['tangent_shuffle_projected']['scores'][i]) and
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
               early_tangent_metric_job={'status':'completed','passing_candidates':passed,'report_sha256':public['report_sha256']})
    append_event(events,'batch_completed',passing_candidates=passed,full_panel_scores=sum(r['local_score'] is not None for f in report['folds'] for r in f['results']))
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
                       early_tangent_metric_job={'status':'failed','error_type':type(exc).__name__,'error':str(exc)})
        raise
