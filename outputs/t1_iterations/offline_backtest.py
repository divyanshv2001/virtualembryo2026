"""Independent early-stage forecast selection, with a once-used later-stage test.

Scores are calibrated to this external atlas, NEVER the hidden T1 validation board.
"""
import importlib.util
import argparse
import json
import sys
import tarfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from iterate import append_event, now
sys.path.insert(0, str(HERE.parents[1]/'outputs/t1_run'))
from run_t1 import digest


def load_core():
    vendor = HERE/'private/scorer_source'
    manifest = json.loads((vendor/'manifest.json').read_text())
    record = next(r for r in manifest['files'] if r['path']=='common/core_metrics.py')
    path = vendor/record['path']
    if digest(path) != record['sha256']: raise ValueError('Scorer source changed')
    spec = importlib.util.spec_from_file_location('pinned_core_metrics', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module, manifest


class Panel:
    """Cache target-only evaluation quantities; never pass them to a learner."""
    weights = {'de_score': .25, 'de_direction': .25, 'mmd_u': .3, 'variogram': .2}
    def __init__(self, core, truth, reference, seed):
        self.core, self.truth, self.reference, self.seed = core, truth, reference, seed
        self.up, self.down, self.dt = core.de_genes(truth, reference)
        self.ref_mean = reference.mean(0, dtype=np.float64)
        # The pinned scorer densifies full gene matrices before averaging in source dtype.
        self.ref_mean_source = reference.mean(0)
        self.truth_mean = truth.mean(0)
        self.chance = max(core._signed_overlap(self.ref_mean_source, self.up, self.down)[0],
            core._signed_overlap(-self.ref_mean_source, self.up, self.down)[0])
        self.pca = PCA(n_components=min(30, truth.shape[0]-1, truth.shape[1]), random_state=0).fit(truth)
        self.b = self.pca.transform(truth)
        distances = ((self.b[:, None]-self.b[None, :])**2).sum(-1)
        self.gamma = 1/(np.median(distances[distances>0])+1e-9)
        rng = np.random.default_rng(seed)
        # Draw the same 20k ordered gene pairs for all candidates in this local panel.
        i = rng.integers(0, truth.shape[1], 20000); j = rng.integers(0, truth.shape[1], 20000)
        self.pairs = (i[i!=j], j[i!=j])
        self.vtruth = self.variogram(truth)

    def variogram(self, x):
        i, j = self.pairs
        # Pair batching changes allocation only, not the variogram estimator.
        return np.concatenate([(np.abs(x[:,i[s:s+1000]]-x[:,j[s:s+1000]])**.5).mean(0)
            for s in range(0, len(i), 1000)])

    def metrics(self, x):
        dp = x.mean(0)-self.ref_mean_source
        dt = self.truth_mean-self.ref_mean_source
        if len(self.up)+len(self.down) == 0 or not np.isfinite(self.chance) or self.chance>=1:
            des = float('nan')
        elif np.std(dp)<.01*np.std(dt): des = 0.
        else:
            overlap = self.core._signed_overlap(dp, self.up, self.down)[0]
            des = (overlap-self.chance)/(1-self.chance)
        a = self.pca.transform(x); na, nb = len(a), len(self.b)
        mmd = 0.
        for scale in [.25,.5,1,2,4]:
            aa = rbf_kernel(a, a, self.gamma*scale); bb = rbf_kernel(self.b, self.b, self.gamma*scale)
            ab = rbf_kernel(a, self.b, self.gamma*scale)
            np.fill_diagonal(aa,0); np.fill_diagonal(bb,0)
            mmd += aa.sum()/(na*(na-1))+bb.sum()/(nb*(nb-1))-2*ab.mean()
        return {'de_score': float(des), 'de_direction': self.core.de_direction(x,self.truth,self.reference),
            'mmd_u': float(mmd/5), 'variogram': float(((self.variogram(x)-self.vtruth)**2).mean())}

    def aggregate(self, raw, floor, ceiling):
        invalid = [k for k in self.weights if not np.isfinite([raw[k],floor[k],ceiling[k]]).all()
            or (floor[k]-ceiling[k] if k in ['mmd_u','variogram'] else ceiling[k]-floor[k]) <= 1e-12]
        if invalid:
            return {'local_score':None,'skills':None,'calibration_valid':False,
                'invalid_calibration_metrics':invalid}
        skills = {k:self.core.skill(raw[k],floor[k],ceiling[k],lower_is_better=k in ['mmd_u','variogram']) for k in self.weights}
        valid = all(np.isfinite(list(skills.values())))
        total = sum(w*(skills[k] if np.isfinite(skills[k]) else 0) for k,w in self.weights.items())*100
        return {'local_score':float(total), 'skills':skills, 'calibration_valid':valid}


def predict(early, last, config, horizon_ratio=1.):
    """Only earlier snapshots and the fixed config enter prediction generation."""
    strength = config.get('strength',0.)*horizon_ratio
    if config['method']=='copy_last': return last.copy()
    if config['method']=='mean_shift':
        return np.maximum(last+strength*(last.mean(0)-early.mean(0)),0).astype(np.float32)
    if config['method']=='quantile_drift':
        q = np.linspace(0,1,33)
        qa = np.quantile(early,q,axis=0); qb = np.quantile(last,q,axis=0)
        future = np.maximum(qb+strength*(qb-qa),0)
        # Project back to monotone quantiles, since extrapolation can cross ranks.
        future = np.maximum.accumulate(future,axis=0)
        ranks = (rankdata(last,axis=0,method='average')-.5)/len(last)
        result = np.empty_like(last)
        for g in range(last.shape[1]): result[:,g] = np.interp(ranks[:,g],q,future[:,g])
        return result
    if config['method']=='local_mean_drift':
        joined = np.vstack([early,last])
        variance = joined.var(0)
        selected = np.argsort(-variance)[:min(512,len(variance))]
        center = joined[:,selected].mean(0); sd = np.maximum(joined[:,selected].std(0),.1)
        scaled = np.clip((joined[:,selected]-center)/sd,-10,10)
        z = PCA(n_components=min(16,len(early)-1),random_state=0).fit_transform(scaled)
        za, zb = z[:len(early)], z[len(early):]
        _, ia = cKDTree(za).query(zb,k=min(16,len(early)))
        _, ib = cKDTree(zb).query(zb,k=min(17,len(last)))
        result = np.empty_like(last)
        for i in range(len(last)):
            neighbors = ib[i][ib[i]!=i][:16]
            delta = last[neighbors].mean(0)-early[ia[i]].mean(0)
            result[i] = np.maximum(last[i]+strength*delta,0)
        return result
    raise ValueError('Unknown model')


def read_atlas():
    path = HERE/'private/early_atlas'
    manifest = json.loads((path/'manifest.json').read_text())
    for r in manifest['files']:
        if digest(path/r['file'])!=r['sha256']: raise ValueError('External source changed')
    metadata = pd.read_csv(path/'metadata.txt',sep=r'\s+')
    with tarfile.open(path/'counts.gz','r:gz') as archive:
        members = archive.getmembers()
        if len(members)!=1 or members[0].name!='counts.txt' or not members[0].isfile(): raise ValueError('Unexpected archive')
        # Read one named member in memory; do not extract archive paths onto disk.
        counts = pd.read_csv(archive.extractfile(members[0]),sep=r'\s+',index_col=0)
    if counts.columns.duplicated().any() or counts.index.duplicated().any(): raise ValueError('Duplicate atlas IDs')
    if set(counts.columns)!=set(metadata.cellName): raise ValueError('Cell metadata mismatch')
    metadata = metadata.set_index('cellName').loc[counts.columns]
    mapping = {'E6.5':6.5,'PS':7.,'NP':7.5,'HF':7.75}
    stages = metadata.embryoStage.map(mapping).to_numpy()
    if np.isnan(stages).any(): raise ValueError('Undeclared stage')
    expected = {6.5:501,7.:138,7.5:259,7.75:307}
    if {k:int((stages==k).sum()) for k in expected}!=expected: raise ValueError('Unexpected stage counts')
    x = counts.to_numpy(dtype=np.float32).T.copy()
    if not np.isfinite(x).all() or (x<0).any(): raise ValueError('Invalid counts')
    library = x.sum(1,dtype=np.float64)
    if (library<=0).any(): raise ValueError('Empty cell')
    x = np.log1p(x*(10000/library[:,None])).astype(np.float32)
    return x, stages, metadata, counts.index.to_numpy(), manifest


def split_embryos(metadata, stage_rows, seed):
    if metadata.iloc[stage_rows].embryo.isna().any(): raise ValueError('Unknown embryo cannot form a validation group')
    ids = metadata.iloc[stage_rows].embryo.to_numpy(dtype=str)
    embryos = np.unique(ids)
    shuffled = np.random.default_rng(seed).permutation(embryos)
    first = shuffled[:len(shuffled)//2]
    mask = np.isin(ids,first)
    return stage_rows[mask], stage_rows[~mask], sorted(first.tolist()), sorted(shuffled[len(shuffled)//2:].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round',default='offline_backtest_02_correction')
    args = parser.parse_args()
    if not args.round.replace('_','').isalnum(): raise ValueError('Invalid round')
    out = HERE/'private'/args.round
    if out.exists(): raise SystemExit('Backtest exists; preserve blind-test history.')
    out.mkdir(parents=True)
    events = out/'events.jsonl'
    configs = [{'name':'copy_last','method':'copy_last'}]
    configs += [{'name':f'{m}_{a}', 'method':m, 'strength':a}
        for m in ['mean_shift','quantile_drift','local_mean_drift'] for a in [.25,.5,1.]]
    plan = {'created_before_execution_utc':now(),'code_sha256':digest(Path(__file__)),
        'selection_fit_stages':[6.5,7.], 'selection_target':7.5,
        'final_fit_stages':[7.,7.5], 'final_target':7.75, 'final_horizon_ratio':.5,
        'configs':configs,'development_panel_seeds':[0,1,2], 'final_panel_seed':20260928,
        'selection':'Highest mean development score among candidates with valid calibration; final target evaluated once after choice.',
        'calibration':'Each local panel gets its own copy-last floor and disjoint-embryo target ceiling.',
        'scope':'External early-stage forecast backtest; not a preview or estimate of the hidden E10.5 leaderboard.',
        'threshold':72., 'submissions_allowed':0, 'jev_requests_allowed':0,
        'correction':'Reject ceilings worse than floor; exclude one NP cell with unknown embryo from validation groups.',
        'final_test_status':'Technical correction after initial test read; not a new untouched holdout. No new models or strengths added.'}
    (out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    (out/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    append_event(events,'plan_frozen',sha256=digest(out/'plan.json'))
    core, scorer_source = load_core()
    x, stages, metadata, genes, atlas_manifest = read_atlas()
    early, last = x[stages==6.5], x[stages==7.]
    # Freeze all development predictions BEFORE fitting any target evaluation quantities.
    predictions = {c['name']:predict(early,last,c) for c in configs}
    append_event(events,'development_predictions_frozen',models=list(predictions),genes=len(genes))
    records = []
    for seed in plan['development_panel_seeds']:
        eligible = (stages==7.5) & metadata.embryo.notna().to_numpy()
        truth_rows, ceiling_rows, truth_embryos, ceiling_embryos = split_embryos(metadata,np.flatnonzero(eligible),seed)
        panel = Panel(core,x[truth_rows],last,seed)
        floor = panel.metrics(last); ceiling = panel.metrics(x[ceiling_rows])
        for config in configs:
            raw = panel.metrics(predictions[config['name']])
            records.append({'candidate':config['name'],'seed':seed,'raw_metrics':raw,
                **panel.aggregate(raw,floor,ceiling)})
        append_event(events,'development_panel_completed',seed=seed,truth_cells=len(truth_rows),
            truth_embryos=truth_embryos,ceiling_embryos=ceiling_embryos)
    summary = []
    for config in configs:
        rows = [r for r in records if r['candidate']==config['name']]
        scores = [r['local_score'] for r in rows if r['calibration_valid']]
        summary.append({'config':config,'mean_local_score':float(np.mean(scores)) if scores else None,
            'sd_local_score':float(np.std(scores,ddof=1)) if len(scores)>1 else None,
            'minimum_local_score':min(scores) if scores else None,
            'valid_panels':len(scores),'required_panels':len(rows),
            'all_calibrations_valid':all(r['calibration_valid'] for r in rows)})
    valid = [r for r in summary if r['all_calibrations_valid']]
    if not valid:
        report = {'plan':plan,'source_manifest':atlas_manifest,'scorer_source':scorer_source,
            'atlas_shape':list(x.shape),'development':records,'summary':summary,'selected':None,
            'status':'calibration_failed','local_threshold_reached':False,'official_72_verified':False,
            'submissions_used':0,'final_once_used_test':None,
            'reason':'At least one required panel has a ceiling not better than the floor. Do not drop metrics or panels to manufacture a passing score.',
            'final_test_history':'First implementation already evaluated E7.75; its reported 60 was invalid because the variogram ceiling was worse than floor.'}
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        append_event(events,'calibration_rejected',local_threshold_reached=False,official_72_verified=False)
        print(json.dumps({'status':report['status'],'summary':summary,'submissions_used':0}),flush=True)
        return
    selected = max(valid,key=lambda r:r['mean_local_score'])
    (out/'selection.json').write_text(json.dumps({'selected':selected,'development_summary':summary,
        'final_target_not_yet_evaluated':True},indent=2),encoding='utf-8')
    append_event(events,'candidate_selected_before_final_evaluation',candidate=selected['config']['name'],mean_local_score=selected['mean_local_score'])
    # The later target enters the evaluator only AFTER configuration selection.
    final_early, final_last = x[stages==7.], x[stages==7.5]
    final_prediction = predict(final_early,final_last,selected['config'],horizon_ratio=.5)
    truth_rows, ceiling_rows, truth_embryos, ceiling_embryos = split_embryos(metadata,np.flatnonzero(stages==7.75),plan['final_panel_seed'])
    panel = Panel(core,x[truth_rows],final_last,plan['final_panel_seed'])
    floor = panel.metrics(final_last); ceiling = panel.metrics(x[ceiling_rows])
    raw = panel.metrics(final_prediction)
    final = {**panel.aggregate(raw,floor,ceiling),'raw_metrics':raw,
        'floor':floor,'ceiling':ceiling,'truth_embryos':truth_embryos,'ceiling_embryos':ceiling_embryos}
    report = {'plan':plan,'source_manifest':atlas_manifest,'scorer_source':scorer_source,
        'atlas_shape':list(x.shape),'development':records,'summary':summary,'selected':selected,'final_once_used_test':final,
        'local_threshold_reached':bool(selected['minimum_local_score']>72 and final['calibration_valid'] and final['local_score']>72),
        'official_72_verified':False,'submissions_used':0,
        'limitations':['Early SMART-seq/FACS atlas differs from later-stage challenge RNA.',
            'Development panels reuse one held-out stage; their spread is sensitivity, not independent replicate performance.',
            'Local anchors and sample sizes differ from official board; local 72 cannot establish official 72.',
            'No new competition prediction written from this backtest.']}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    append_event(events,'final_test_completed',local_score=final['local_score'],local_threshold_reached=report['local_threshold_reached'],official_72_verified=False)
    print(json.dumps({'summary':summary,'final_local_score':final['local_score'],'submissions_used':0}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2): main()
