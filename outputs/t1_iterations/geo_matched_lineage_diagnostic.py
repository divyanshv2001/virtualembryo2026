"""Cached stage/region/lineage rank diagnostic with fixed nuisance shuffles."""
import gzip
import json
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event


def region_group(region):
    if 'atrium' in region: return 'atrium'
    if 'ventric' in region: return 'ventricle'
    return region


def residual_ranks(values, strata):
    ranks=rankdata(values)
    for value in np.unique(strata):
        mask=strata==value
        ranks[mask]-=ranks[mask].mean()
    return ranks


def correlation(a,b):
    denom=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denom) if denom else 0.


def main():
    out=HERE/'private/geo_matched_lineage_diagnostic_01'
    if out.exists(): raise ValueError('Preserve existing diagnostic')
    compatibility_path=HERE/'GSE76118_EARLY_COMPATIBILITY_RESULTS.json'
    cohort_path=HERE/'GSE76118_EARLY_COHORT_RESULTS.json'
    mapping_path=HERE/'private/geo_early_prospective_policy_01/stable_id_panel_mapping.json'
    prior=json.loads(compatibility_path.read_text())
    cohort={r['gsm']:r for r in json.loads(cohort_path.read_text())['outcomes']}
    selected=[dict(r,canonical_region=region_group(r['region'])) for r in prior['outcomes']
              if r['diagnostic_nearest_group']=='cardiomyocyte']
    composition=Counter((r['stage'],r['canonical_region']) for r in selected)
    regions=sorted(g for g in {r['canonical_region'] for r in selected}
                   if all(composition[(s,g)]>=2 for s in ('E8.5','E9.5')))
    source=HERE/'private/associated_prepared_01'
    source_report=json.loads((source/'report.json').read_text())
    for filename,key in [('expression.npy','expression_sha256'),('genes.csv','genes_sha256'),('selected_metadata.csv','metadata_sha256')]:
        if digest(source/filename)!=source_report[key]:raise ValueError('Atlas checksum mismatch')
    genes=pd.read_csv(source/'genes.csv').fillna('')
    freq=Counter(genes.symbol)
    lookup={r.symbol:i for i,r in enumerate(genes.itertuples()) if r.symbol and freq[r.symbol]==1}
    mapping=json.loads(mapping_path.read_text())
    with gzip.open(HERE/'private/geo_early_schema_audit_01/GSM2035425.HTSeq.output.txt.gz','rt') as f:
        external_ids={line.split('\t')[0] for line in f if not line.startswith('__')}
    ids=sorted(g for g,s in mapping.items() if g in external_ids and s in lookup)
    columns=np.array([lookup[mapping[g]] for g in ids])
    meta=pd.read_csv(source/'selected_metadata.csv');stages=meta.numeric_stage.to_numpy(float)
    cm=meta.celltype_extended_atlas.str.startswith('Cardiomyocytes').to_numpy()
    refs={stage:np.flatnonzero(cm&(stages==stage)) for stage in (8.5,9.5)}
    plan={'created_utc':now(),'code_sha256':digest(HERE/'geo_matched_lineage_diagnostic.py'),
        'compatibility_sha256':digest(compatibility_path),'cohort_sha256':digest(cohort_path),
        'mapping_sha256':digest(mapping_path),'atlas_report_sha256':digest(source/'report.json'),
        'seed':20261001,'shuffles':64,'shared_genes':len(ids),
        'selected_gsm_ids':[r['gsm'] for r in selected], 'region_composition':{str(k):v for k,v in composition.items()},
        'overlap_regions_with_at_least_two_cells_per_stage':regions,
        'region_rule':'Atrium groups by atrium substring; ventricle/ventricular septum by ventric substring. Broad regional matching only, not proof of anatomical homology.',
        'primary':'Equal-weight overlapping broad-region external CM-like delta versus atlas annotated cardiomyocyte delta. External identity is E8.5-frozen diagnostic only.',
        'nuisance_control':'E8.5-only atlas CM mean rank deciles and detection rank quintiles, stable gene-index ties; shuffle external delta within fixed50strata.',
        'advance_gate':'At least2overlapregions, primarystratifiedrankcorrelation>.05 and empiricalone-sidedshufflep<=.05. Passing authorizes further diagnostics only, not forecast/promotion.',
        'scope':'E8.5->E9.5 compatibility only. E9.5 external/atlas reads here cannot train an E8.5/E9.5 validation forecast. No predictive fullpanel metric, reward or independent embryo validation.'}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (out/'executed_source.py').write_bytes((HERE/'geo_matched_lineage_diagnostic.py').read_bytes())
    events=out/'events.jsonl';append_event(events,'plan_frozen',plan_sha256=digest(out/'plan.json'))
    x=np.load(source/'expression.npy',mmap_mode='r');means={s:np.empty(len(ids)) for s in refs};detection=np.empty(len(ids))
    for start in range(0,len(ids),256):
        sl=slice(start,min(start+256,len(ids)))
        for stage,rows in refs.items():
            v=np.asarray(x[np.ix_(rows,columns[sl])],dtype=np.float32)
            means[stage][sl]=v.mean(0,dtype=np.float64)
            if stage==8.5:detection[sl]=(v>0).mean(0)
    def bins(values,n):
        order=np.argsort(values,kind='stable');result=np.empty(len(values),dtype=int)
        result[order]=np.arange(len(values))*n//len(values);return result
    strata=5*bins(means[8.5],10)+bins(detection,5)
    np.savez_compressed(out/'fixed_strata.npz',gene_ids=np.array(ids),strata=strata)
    append_event(events,'strata_frozen',sha256=digest(out/'fixed_strata.npz'))
    data={}
    for cell in selected:
        info=cohort[cell['gsm']]
        path=HERE/('private/geo_early_schema_audit_01' if info['reused_pilot'] else 'private/geo_early_cohort_audit_01')/(cell['gsm']+'.HTSeq.output.txt.gz')
        if digest(path)!=info['file_sha256']:raise ValueError('External checksum changed')
        with gzip.open(path,'rt') as f:counts={g:float(v) for g,v in (line.rstrip().split('\t') for line in f) if not g.startswith('__')}
        data[cell['gsm']]=np.log1p(10000*np.array([counts[g] for g in ids])/sum(counts.values()))
    def average(stage,region=None):
        cells=[data[c['gsm']] for c in selected if c['stage']==stage and (region is None or c['canonical_region']==region)]
        return np.mean(cells,axis=0)
    deltas={'cm_pooled':average('E9.5')-average('E8.5')}
    if regions:
        for region in regions:deltas['region_'+region]=average('E9.5',region)-average('E8.5',region)
        deltas['region_balanced']=np.mean([deltas['region_'+r] for r in regions],axis=0)
    atlas_delta=means[9.5]-means[8.5];atlas_rank=residual_ranks(atlas_delta,strata)
    strata_rows=[np.flatnonzero(strata==v) for v in np.unique(strata)]
    results={};rng=np.random.default_rng(20261001)
    for name,delta in deltas.items():
        rank=residual_ranks(delta,strata);observed=correlation(rank,atlas_rank);null=[]
        for _ in range(64):
            shuffled=rank.copy()
            for rows in strata_rows:shuffled[rows]=rng.permutation(rank[rows])
            null.append(correlation(shuffled,atlas_rank))
        results[name]={'pooled_spearman':float(spearmanr(delta,atlas_delta).statistic),
            'stratified_rank_correlation':observed,'shuffle_correlations':null,
            'one_sided_empirical_p':float((1+sum(v>=observed for v in null))/65),
            'shuffle_95_percentile':float(np.quantile(null,.95))}
    primary=results.get('region_balanced',{})
    passes=len(regions)>=2 and primary.get('stratified_rank_correlation',-1)>.05 and primary.get('one_sided_empirical_p',1)<=.05
    report={'updated_utc':now(),'plan_sha256':digest(out/'plan.json'),'results':results,
        'region_composition':plan['region_composition'],'supported_broad_regions':regions,
        'external_cm_like_cells':len(selected),'atlas_cm_cells':{str(s):len(v) for s,v in refs.items()},
        'diagnostic_gate_passed':passes,'temporal_prior_validated':False,
        'models_fit':0,'full_panel_scores':0,'reward_delta':0,
        'limitations':'Small external cohort, inferred identities, broad tissue matching and unresolved embryo/capture/assay effects. Shuffling genes does not provide biological replication or a predictive score.',
        'next_action':'Further independent validation required' if passes else 'Reject this external pooled/matched temporal prior; next source-only saturating-time formula proxy with frozen matched controls.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    public=dict(report,report_sha256=digest(out/'report.json'))
    (HERE/'GSE76118_MATCHED_LINEAGE_RESULTS.json').write_text(json.dumps(public,indent=2)+'\n')
    append_event(events,'matched_diagnostic_completed',report_sha256=public['report_sha256'],gate_passed=passes)
    print(json.dumps({k:v for k,v in public.items() if k!='results'}))
    print(json.dumps({k:{a:b for a,b in v.items() if a!='shuffle_correlations'} for k,v in results.items()}))


if __name__=='__main__':main()
