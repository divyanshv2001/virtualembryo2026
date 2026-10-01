"""Audit actual local scorer versus ranking screens; no forecast evaluation."""
import json
import numpy as np
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event
from offline_backtest import load_core, Panel
from eb_gene_slope_screen import score, K


def main():
    out=HERE/'private/proxy_fidelity_audit_01'
    if out.exists():raise ValueError('Preserve earlier audit')
    core,manifest=load_core()
    dependencies=['offline_backtest.py','eb_gene_slope_screen.py','joint_detection_coupling_temporal.py']
    prior_path=HERE/'private/joint_detection_coupling_temporal_01/report.json'
    prior=json.loads(prior_path.read_text())
    plan={'created_utc':now(),'code_sha256':digest(HERE/'proxy_fidelity_audit.py'),
        'dependency_sha256':{n:digest(HERE/n) for n in dependencies},
        'scorer_manifest_sha256':digest(HERE/'private/scorer_source/manifest.json'),
        'frozen_fullpanel_report_sha256':digest(prior_path),'seed':20261001,
        'scope':'Source-code and frozen-plan audit plus small synthetic no-effect fixture. No learned forecasts, full-panel score batch or official calibration.'}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (out/'executed_source.py').write_bytes((HERE/'proxy_fidelity_audit.py').read_bytes())
    append_event(out/'events.jsonl','plan_frozen',plan_sha256=digest(out/'plan.json'))
    rng=np.random.default_rng(20261001)
    reference=rng.lognormal(-1,.5,(80,1000)).astype(np.float32)
    truth=rng.lognormal(-1,.5,(80,1000)).astype(np.float32)
    truth[:,:70]+=.8;truth[:,70:100]=np.maximum(0,truth[:,70:100]-.4)
    up,down,_=core.de_genes(truth,reference)
    fixture_ref=reference.mean(0,dtype=np.float64)
    fixture_delta=truth.mean(0,dtype=np.float64)-fixture_ref
    order=np.argsort(fixture_delta,kind='stable')
    proxy_copy=score(np.zeros(1000),fixture_delta,fixture_ref,order[-K:],order[:K])
    core_copy=core.de_score(reference,truth,reference)
    if core_copy['score']!=0 or core.de_direction(reference,truth,reference)!=0:
        raise ValueError('Pinned no-change fixture invariant failed')
    differences=[
        {'aspect':'horizon','proxy':'Quarterday target increments','full_scorer_workflow':'Frozen one-day8->9 and8.25->9.25; official E9.5->E10.5. Nonlinear dynamics can change ranking with horizon.'},
        {'aspect':'DE truth sets','proxy':'Fixed200up/200down by mean delta, no significance/effect filter','full_scorer_workflow':'Mann-Whitney/BHalpha.05, mean log shift>=.25 and min10cells; unequal variable up/down counts.'},
        {'aspect':'gene panel','proxy':'26775unique atlas-mapped genes (external static25464support)','full_scorer_workflow':'Unchanged32285panel, prior localsource adapter zero-fills unavailable mapping identically for candidates/controls; limits challenge transfer.'},
        {'aspect':'no-effect guard','proxy':'No relative-delta std guard; copy may have negative overlap due ranking ties','full_scorer_workflow':'DE/DCS return0 when predicted delta std<.01truthdelta std; copy rawfloor0, skill50.'},
        {'aspect':'cells and float precision','proxy':'All preparedstage rows, float64mean delta, stableargsort','full_scorer_workflow':'Seeded1500donors,1000truth/1000ceiling; scorer source-dtype mean and native overlap sort ties.'},
        {'aspect':'distribution metrics','proxy':'Neither MMD nor variogram','full_scorer_workflow':'MMD.30+variogram.20 headline weight; frozen target PCA/bandwidth/pairs. Mean vectors alone cannot assess them.'},
        {'aspect':'calibration','proxy':'Chance-adjusted overlap and rawpartialrank only, no metricfloor/ceiling or aggregate','full_scorer_workflow':'Pinned skillfloor50/ceiling100 hyperbolic map; metricgap validation and weighted4metric aggregate.'},
        {'aspect':'candidate correspondence','proxy':'Recentlinear and formula responsevectors','full_scorer_workflow':'Copy/anchorCNF/coupling whole-cell forecasts; no nontrivial exactly matched candidate/split/horizon pair currently available.'}]
    report={'updated_utc':now(),'plan_sha256':digest(out/'plan.json'),'differences':differences,
        'fullpanel_weights':Panel.weights,'proxy_K':K,
        'frozen_fullpanel_folds':prior['plan']['folds'],
        'frozen_fullpanel_candidate_names':prior['plan']['candidates'],
        'nontrivial_matched_proxy_fullpanel_pairs':0,'empirical_proxy_headline_correlation':None,
        'synthetic_unit_probe':{'cells_per_group':80,'genes':1000,'selected_core_up':len(up),'selected_core_down':len(down),
             'core_copy_raw_DE':core_copy['score'],'core_copy_raw_direction':0,
             'proxy_copy_overlap':proxy_copy['chance_adjusted_overlap'],
             'interpretation':'Synthetic code probe only, not a benchmark, learned prediction, skill, or leaderboard score.'},
        'models_fit':0,'full_panel_scores':0,'reward_delta':0,'scorer_changed':False,
        'conclusion':'Proxy rejections are heuristic screens, not proof of one-day four-metric failure. No measured proxy fidelity; do not retune proxy gates to known outcomes.',
        'next_experiment':'Freeze a bounded10score full-panel one-day ablation on existing sourcefolds: persistence, archived .5/.5 anchor incumbent, anchored recentpooled slope, lineageglobal slope and partialpool1000 slope. Identical1500donors/1000truth+1000ceiling, seed20260928, scorer/calibration and mappedmass safeguards. Pastfits and forecastsfrozen beforetarget read. Use fixed conservative horizon/shrinkage, no quarterdaygate requirement; development only.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    public=dict(report,report_sha256=digest(out/'report.json'))
    (HERE/'PROXY_SCORER_HORIZON_AUDIT.json').write_text(json.dumps(public,indent=2)+'\n')
    append_event(out/'events.jsonl','audit_completed',report_sha256=public['report_sha256'])
    print(json.dumps({k:v for k,v in public.items() if k!='differences'}))


if __name__=='__main__':main()
