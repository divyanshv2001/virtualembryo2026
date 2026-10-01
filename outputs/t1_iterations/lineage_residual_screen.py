"""Past-only lineage standardization and two-part odds historical proxy screen.

Source annotations are supplied atlas labels, not independently validated
lineages. No challenge reads or full-panel score; no research reward earned.
"""
import json
from collections import Counter

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event
from eb_gene_slope_screen import score, K

FOLDS = [(8., 8.25), (8.25, 8.5)]
ALPHAS = [.25, .5]
MIN_CELLS = 20


def lineage(label):
    text = str(label).lower()
    if 'cardiomyocyte' in text:
        return 'cardiomyocyte'
    if 'cardiopharyngeal' in text or 'pharyngeal mesoderm' in text:
        return 'cardiopharyngeal'
    if 'endothelium' in text or text == 'endocardium':
        return 'endothelial'
    if any(s in text for s in ('blood', 'erythroid', 'emp', 'haemato', 'megakaryo')):
        return 'blood'
    if any(s in text for s in ('gut', 'endoderm')):
        return 'endoderm'
    if 'neural crest' in text:
        return 'neural_crest'
    if any(s in text for s in ('mesenchyme', 'mesoderm', 'mesothelium', 'epicardium')):
        return 'mesenchymal'
    return 'ectoderm_other'


def main():
    source = HERE/'private/associated_prepared_01'
    out = HERE/'private/lineage_residual_screen_01'
    if out.exists():
        raise ValueError('Preserve original frozen run; do not overwrite')
    manifest = json.loads((source/'report.json').read_text())
    for filename, key in [('expression.npy', 'expression_sha256'),
                          ('selected_metadata.csv', 'metadata_sha256'), ('genes.csv', 'genes_sha256')]:
        if digest(source/filename) != manifest[key]:
            raise ValueError('Prepared input hash mismatch: '+filename)
    panel_path = HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt'
    panel = panel_path.read_text().splitlines()
    symbols = pd.read_csv(source/'genes.csv').symbol.fillna('').tolist()
    counts = Counter(symbols)
    lookup = {s:i for i,s in enumerate(symbols) if s and counts[s] == 1}
    mapped = np.array([i for i,s in enumerate(panel) if s in lookup])
    atlas = np.array([lookup[panel[i]] for i in mapped])
    plan = {'created_utc':now(), 'code_sha256':digest(HERE/'lineage_residual_screen.py'),
            'proxy_code_sha256':digest(HERE/'eb_gene_slope_screen.py'),
            'source_manifest_sha256':digest(source/'report.json'),
            'panel_sha256':digest(panel_path), 'folds':FOLDS, 'alphas':ALPHAS,
            'minimum_cells_per_group_per_stage':MIN_CELLS,
            'lineages':'Fixed label mapping in code; supplied source atlas annotations, no target-label fitting. Not independent lineage ground truth.',
            'mechanisms': ['Current-composition-weighted within-lineage mean slope',
                           'Current-composition-weighted detection log-odds slope plus conditional-positive slope'],
            'odds_regularizer': 'Jeffreys half-count; conditional-positive means multiplied by factor clipped .5..2; zero-detection groups remain zero.',
            'missing_support':'Groups below20 cells in either earlier stage persist; no dropping/renormalizing supported weights.',
            'prediction':'Blend pooled recent mean delta with frozen within-lineage delta at alpha .25/.5; no future composition prediction.',
            'screen':'Both folds strictly better chance-adjusted signed-top200 overlap than pooled recent-linear and no partial-Spearman regression. Require both > persistence direction too.',
            'limitations':'Mapped-gene rank proxy only; no full-panel metrics, challenge fitting, independent embryos or reward. Lineage correction alone is not latent-state-adjusted residual regression.'}
    out.mkdir()
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    events = out/'events.jsonl'
    append_event(events,'plan_frozen',plan_sha256=digest(out/'plan.json'))
    x = np.load(source/'expression.npy',mmap_mode='r')
    metadata = pd.read_csv(source/'selected_metadata.csv')
    stages = metadata.numeric_stage.to_numpy(float)
    labels = metadata.celltype_extended_atlas.map(lineage).to_numpy()
    groups = sorted(set(labels[stages<=8.]))

    def statistics(stage, use_lineages):
        rows = np.flatnonzero(stages==stage)
        mean = np.zeros(len(atlas))
        grouped = {}
        for start in range(0,len(atlas),256):
            sl = slice(start,min(start+256,len(atlas)))
            v = np.asarray(x[np.ix_(rows,atlas[sl])],dtype=np.float32)
            mean[sl] = v.mean(0,dtype=np.float64)
            if use_lineages:
                for group in groups:
                    selected = labels[rows]==group
                    n = int(selected.sum())
                    if group not in grouped:
                        grouped[group] = {'n':n,'p':np.zeros(len(atlas)),
                                          'mu':np.zeros(len(atlas)), 'mean':np.zeros(len(atlas))}
                    if n:
                        a = v[selected]
                        detected = (a>0).sum(0)
                        total = a.sum(0,dtype=np.float64)
                        grouped[group]['p'][sl] = (detected+.5)/(n+1.)
                        grouped[group]['mu'][sl] = total/np.maximum(detected,1)
                        grouped[group]['mean'][sl] = total/n
        return {'n':len(rows),'mean':mean,'groups':grouped}

    history = {}; outcomes=[]; deltas={}
    for cutoff,target in FOLDS:
        for stage in (cutoff-.25,cutoff):
            if stage not in history:
                history[stage] = statistics(stage,True)
        earlier,current = history[cutoff-.25],history[cutoff]
        ref=current['mean']; baseline=ref-earlier['mean']
        within=np.zeros_like(ref); odds=np.zeros_like(ref); supported=[]
        for group in groups:
            a,b=earlier['groups'][group],current['groups'][group]
            if min(a['n'],b['n'])<MIN_CELLS:
                continue
            supported.append(group)
            weight=b['n']/current['n']
            within += weight*(b['mean']-a['mean'])
            pn=expit(2*logit(b['p'])-logit(a['p']))
            proposed=np.maximum(0,2*b['mu']-a['mu'])
            mun=b['mu']*np.clip(proposed/np.maximum(b['mu'],1e-12),.5,2.)
            # Use observed zero fraction in current mean; shrinkage affects only the change.
            odds += weight*(pn-b['p'])*mun + weight*b['p']*(mun-b['mu'])
        candidates={'copy':np.zeros_like(ref),'recent_linear':baseline}
        for alpha in ALPHAS:
            candidates[f'within_a{alpha}']=(1-alpha)*baseline+alpha*within
            candidates[f'odds_a{alpha}']=(1-alpha)*baseline+alpha*odds
        for name,delta in candidates.items():
            deltas[f'{cutoff}_{name}']=delta.astype(np.float32)
        frozen=out/f'cutoff_{cutoff}_deltas.npz'
        np.savez_compressed(frozen,mapped=mapped,**{k:v.astype(np.float32) for k,v in candidates.items()})
        append_event(events,'predictions_frozen_before_target_read',cutoff=cutoff,target=target,
                     prediction_sha256=digest(frozen),supported_groups=supported)
        future=statistics(target,False)
        truth=future['mean']-ref
        order=np.argsort(truth,kind='stable');up,down=order[-K:],order[:K]
        results={name:score(delta,truth,ref,up,down) for name,delta in candidates.items()}
        outcomes.append({'cutoff':cutoff,'target':target,'supported_groups':supported,
                         'current_group_counts':{g:s['n'] for g,s in current['groups'].items()},
                         'results':results})
        append_event(events,'proxy_fold_evaluated',cutoff=cutoff,results=results)
    passing=[name for name in candidates if name not in ('copy','recent_linear') and all(
        f['results'][name]['chance_adjusted_overlap']>f['results']['recent_linear']['chance_adjusted_overlap']
        and f['results'][name]['partial_spearman']>=f['results']['recent_linear']['partial_spearman']
        and f['results'][name]['partial_spearman']>f['results']['copy']['partial_spearman']
        for f in outcomes)]
    report={'status':'completed','outcomes':outcomes,'passing_candidates':passing,
            'full_panel_scores':0,'research_reward_delta':0,'official_submissions':0,
            'scope':plan['limitations'],'plan_sha256':digest(out/'plan.json')}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    public=dict(report,updated_utc=now(),report_sha256=digest(out/'report.json'))
    (HERE/'LINEAGE_RESIDUAL_SCREEN_RESULTS.json').write_text(json.dumps(public,indent=2)+'\n')
    append_event(events,'screen_completed',passing_candidates=passing,report_sha256=public['report_sha256'])
    print(json.dumps(public))


if __name__=='__main__':
    main()
