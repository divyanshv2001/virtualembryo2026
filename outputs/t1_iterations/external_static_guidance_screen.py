"""Past-only external E8.5 static cardiac reliability guidance proxy."""
import gzip
import json
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event
from eb_gene_slope_screen import score, K


def guidance(detection):
    # Average ranks prevent arbitrary gene-ID ordering of detection ties.
    return 2*(rankdata(detection)-.5)/len(detection)-1


def main():
    out=HERE/'private/external_static_guidance_screen_01'
    if out.exists():raise ValueError('Preserve prior screen')
    source=HERE/'private/associated_prepared_01'
    report=json.loads((source/'report.json').read_text())
    for name,key in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(source/name)!=report[key]:raise ValueError('Source changed')
    compatibility_path=HERE/'GSE76118_EARLY_COMPATIBILITY_RESULTS.json'
    compatibility=json.loads(compatibility_path.read_text())
    cells=[r for r in compatibility['outcomes'] if r['stage']=='E8.5' and r['diagnostic_nearest_group']=='cardiomyocyte']
    assert len(cells)==7
    cohort_path=HERE/'GSE76118_EARLY_COHORT_RESULTS.json'
    cohort={r['gsm']:r for r in json.loads(cohort_path.read_text())['outcomes']}
    mapping_path=HERE/'private/geo_early_prospective_policy_01/stable_id_panel_mapping.json'
    mapping=json.loads(mapping_path.read_text())
    genes=pd.read_csv(source/'genes.csv').fillna('');freq=Counter(genes.symbol)
    lookup={r.symbol:i for i,r in enumerate(genes.itertuples()) if r.symbol and freq[r.symbol]==1}
    panel_path=HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt'
    panel=panel_path.read_text().splitlines()
    mapped=np.array([i for i,s in enumerate(panel) if s in lookup]);columns=np.array([lookup[panel[i]] for i in mapped])
    panel_index={panel[i]:j for j,i in enumerate(mapped)}
    prior_map={gid:panel_index[symbol] for gid,symbol in mapping.items() if symbol in panel_index}
    folds=[(8.5,8.75),(8.75,9.)];alphas=[.1,.25]
    plan={'created_utc':now(),'code_sha256':digest(HERE/'external_static_guidance_screen.py'),
        'score_dependency_sha256':digest(HERE/'eb_gene_slope_screen.py'),
        'source_report_sha256':digest(source/'report.json'),'panel_sha256':digest(panel_path),
        'compatibility_sha256':digest(compatibility_path),'cohort_sha256':digest(cohort_path),
        'mapping_sha256':digest(mapping_path),'external_fit_gsm_ids':[r['gsm'] for r in cells],
        'external_fit_stage_only':'E8.5','folds':folds,'alphas':alphas,'seed':20261001,
        'formula':'delta=pooled_recent_delta + alpha*source_current_CM_fraction*source_recent_CM_delta*external_E8.5_detection_rank_centered. Static guidance in [-1,1], average tied ranks; unmapped genes guidance0.',
        'controls':'Copy, recent pooled linear, matched source-current-CM detection guidance, within source-current-expression-decile shuffled external guidance. Same cardiac component, alpha and mapping support.',
        'gate':'Both folds strictly improve signed top200 chance-adjusted overlap vs recent/source-guidance/shuffle; partial Spearman>=all three and >0. No target-based tuning.',
        'scope':'Mapped-gene rank/direction development proxy; repeated source folds are not independent embryo validation. No externalE9.5 expression read, temporal coefficients or forecast training. No fullpanel scores/reward.',
        'resources':'D memmap256gene chunks; save small frozen vectors/split IDs, no full predictions.'}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (out/'executed_source.py').write_bytes((HERE/'external_static_guidance_screen.py').read_bytes())
    events=out/'events.jsonl';append_event(events,'plan_frozen',plan_sha256=digest(out/'plan.json'))
    detection=np.zeros(len(mapped));support=np.zeros(len(mapped),dtype=bool)
    for cell in cells:
        info=cohort[cell['gsm']];assert info['stage']=='E8.5' and info['prospective_qc_pass']
        path=HERE/('private/geo_early_schema_audit_01' if info['reused_pilot'] else 'private/geo_early_cohort_audit_01')/(cell['gsm']+'.HTSeq.output.txt.gz')
        if digest(path)!=info['file_sha256']:raise ValueError('External input changed')
        with gzip.open(path,'rt') as f:
            for line in f:
                gid,value=line.rstrip().split('\t')
                if gid in prior_map:
                    index=prior_map[gid];support[index]=True;detection[index]+=float(value)>0
    external=np.zeros(len(mapped));external[support]=guidance(detection[support]/len(cells))
    np.savez_compressed(out/'external_E8_5_guidance.npz',mapped=mapped,support=support,guidance=external)
    append_event(events,'static_external_prior_frozen',sha256=digest(out/'external_E8_5_guidance.npz'),stage='E8.5',cells=len(cells))
    x=np.load(source/'expression.npy',mmap_mode='r');meta=pd.read_csv(source/'selected_metadata.csv')
    stages=meta.numeric_stage.to_numpy(float);cm=meta.celltype_extended_atlas.str.startswith('Cardiomyocytes').to_numpy()
    outcomes=[]
    for cutoff,target in folds:
        current=np.flatnonzero(stages==cutoff);previous=np.flatnonzero(stages==cutoff-.25)
        ccur=current[cm[current]];cprev=previous[cm[previous]]
        if min(len(ccur),len(cprev))<20:raise ValueError('Insufficient past cardiac cells')
        ref=np.empty(len(mapped));delta=np.empty(len(mapped));cdelta=np.empty(len(mapped));source_detection=np.empty(len(mapped))
        for start in range(0,len(mapped),256):
            sl=slice(start,min(start+256,len(mapped)));cols=columns[sl]
            ref[sl]=np.asarray(x[np.ix_(current,cols)]).mean(0,dtype=np.float64)
            delta[sl]=ref[sl]-np.asarray(x[np.ix_(previous,cols)]).mean(0,dtype=np.float64)
            v=np.asarray(x[np.ix_(ccur,cols)]);source_detection[sl]=(v>0).mean(0)
            cdelta[sl]=v.mean(0,dtype=np.float64)-np.asarray(x[np.ix_(cprev,cols)]).mean(0,dtype=np.float64)
        source_prior=np.zeros(len(mapped));source_prior[support]=guidance(source_detection[support])
        shuffled=external.copy();indices=np.flatnonzero(support)
        ordering=indices[np.argsort(ref[indices],kind='stable')]
        bins=np.arange(len(indices))*10//len(indices);rng=np.random.default_rng(20261001+int(cutoff*100))
        for b in range(10):
            group=ordering[bins==b];shuffled[group]=rng.permutation(external[group])
        component=len(ccur)/len(current)*cdelta
        candidates={'copy':np.zeros_like(delta),'recent_linear':delta}
        for alpha in alphas:
            for name,prior in [('external',external),('source',source_prior),('shuffled',shuffled)]:
                candidates[f'{name}_{alpha}']=delta+alpha*component*prior
        frozen=out/f'deltas_{cutoff}.npz';np.savez_compressed(frozen,mapped=mapped,current=current,previous=previous,**candidates)
        append_event(events,'proxy_predictions_frozen_before_target',cutoff=cutoff,target=target,sha256=digest(frozen))
        target_rows=np.flatnonzero(stages==target);truth=np.empty(len(mapped))
        for start in range(0,len(mapped),256):
            sl=slice(start,min(start+256,len(mapped)))
            truth[sl]=np.asarray(x[np.ix_(target_rows,columns[sl])]).mean(0,dtype=np.float64)-ref[sl]
        order=np.argsort(truth,kind='stable');up,down=order[-K:],order[:K]
        scores={name:score(pred,truth,ref,up,down) for name,pred in candidates.items()}
        row={'cutoff':cutoff,'target':target,'past_cm_counts':[len(cprev),len(ccur)],'cm_fraction':len(ccur)/len(current),'results':scores}
        outcomes.append(row);append_event(events,'development_proxy_scored',**row)
    passing=[]
    for alpha in alphas:
        if all(f['results'][f'external_{alpha}']['chance_adjusted_overlap']>max(f['results'][c]['chance_adjusted_overlap'] for c in ('recent_linear',f'source_{alpha}',f'shuffled_{alpha}'))
            and f['results'][f'external_{alpha}']['partial_spearman']>=max(f['results'][c]['partial_spearman'] for c in ('recent_linear',f'source_{alpha}',f'shuffled_{alpha}'))
            and f['results'][f'external_{alpha}']['partial_spearman']>0 for f in outcomes):passing.append(alpha)
    report={'updated_utc':now(),'plan_sha256':digest(out/'plan.json'),'outcomes':outcomes,
        'passing_alphas':passing,'external_E8_5_cells':7,'external_guidance_supported_genes':int(support.sum()),
        'proxy_evaluations':sum(len(f['results']) for f in outcomes),'full_panel_scores':0,'reward_delta':0,
        'external_E9_5_expression_used':False,'independent_embryo_validation':False,
        'next_action':'Freeze full-panel matched-control ablation for passing estimator' if passing else 'Reject tested static guidance formula; next source-only saturating-time formula proxy. External scientific family remains open.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    public=dict(report,report_sha256=digest(out/'report.json'))
    (HERE/'EXTERNAL_STATIC_GUIDANCE_RESULTS.json').write_text(json.dumps(public,indent=2)+'\n')
    append_event(events,'screen_completed',report_sha256=public['report_sha256'],passing_alphas=passing)
    print(json.dumps(public))


if __name__=='__main__':main()
