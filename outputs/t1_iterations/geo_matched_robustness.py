"""Conditional cell bootstrap and leave-one-out; no predictive benchmark scores."""
import gzip
import json
from collections import Counter

import numpy as np
import pandas as pd
from geo_matched_lineage_diagnostic import residual_ranks, correlation, region_group
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event


def main():
    out=HERE/'private/geo_matched_robustness_01'
    if out.exists():raise ValueError('Keep original robustness run; no overwrite')
    matched_path=HERE/'GSE76118_MATCHED_LINEAGE_RESULTS.json'
    compatibility_path=HERE/'GSE76118_EARLY_COMPATIBILITY_RESULTS.json'
    cohort_path=HERE/'GSE76118_EARLY_COHORT_RESULTS.json'
    strata_path=HERE/'private/geo_matched_lineage_diagnostic_01/fixed_strata.npz'
    matched=json.loads(matched_path.read_text());compat=json.loads(compatibility_path.read_text())
    cohort={r['gsm']:r for r in json.loads(cohort_path.read_text())['outcomes']}
    regions=matched['supported_broad_regions']
    selected=[dict(r,broad_region=region_group(r['region'])) for r in compat['outcomes']
              if r['diagnostic_nearest_group']=='cardiomyocyte' and region_group(r['region']) in regions]
    strata_source=np.load(strata_path);ids=strata_source['gene_ids'].tolist();strata=strata_source['strata']
    source=HERE/'private/associated_prepared_01'
    source_manifest=json.loads((source/'report.json').read_text())
    for name,key in [('expression.npy','expression_sha256'),('genes.csv','genes_sha256'),('selected_metadata.csv','metadata_sha256')]:
        if digest(source/name)!=source_manifest[key]:raise ValueError('Atlas changed')
    plan={'created_utc':now(),'code_sha256':digest(HERE/'geo_matched_robustness.py'),
        'dependency_sha256':digest(HERE/'geo_matched_lineage_diagnostic.py'),
        'input_sha256':{p.name:digest(p) for p in (matched_path,compatibility_path,cohort_path,strata_path)},
        'atlas_report_sha256':digest(source/'report.json'),
        'selected_gsm_ids':[r['gsm'] for r in selected],'regions':regions,'seed_base':2026100101,
        'replicates':64,'bootstrap':'Resample independently within each external stage/broad-region cell group, retaining original group counts and equal region weights; atlas delta and nuisance strata fixed.',
        'leave_one_out':'Remove each selected external cell once; retain all broad regions. No refit of diagnostic labels/strata.',
        'stability_gate':'2.5% cell-bootstrap stratified rho >0 AND every leave-one-cell-out rho >0. No gate retuning after outcomes.',
        'scope':'Conditional uncertainty for these selected cells, not independent embryo replication. No forecast/full-panel metrics/reward/readiness replicates. E9.5 remains compatibility-only for earlier validation.',
        'resources':'D cached expression, atlas memmap256genechunks, external14x25464 matrix under3MB.'}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (out/'executed_source.py').write_bytes((HERE/'geo_matched_robustness.py').read_bytes())
    events=out/'events.jsonl';append_event(events,'plan_frozen',plan_sha256=digest(out/'plan.json'))
    mapping=json.loads((HERE/'private/geo_early_prospective_policy_01/stable_id_panel_mapping.json').read_text())
    genes=pd.read_csv(source/'genes.csv').fillna('');freq=Counter(genes.symbol)
    lookup={r.symbol:i for i,r in enumerate(genes.itertuples()) if r.symbol and freq[r.symbol]==1}
    columns=np.array([lookup[mapping[g]] for g in ids])
    meta=pd.read_csv(source/'selected_metadata.csv');stages=meta.numeric_stage.to_numpy(float)
    cm=meta.celltype_extended_atlas.str.startswith('Cardiomyocytes').to_numpy()
    atlas_delta=np.empty(len(ids));x=np.load(source/'expression.npy',mmap_mode='r')
    a=np.flatnonzero(cm&(stages==8.5));b=np.flatnonzero(cm&(stages==9.5))
    for start in range(0,len(ids),256):
        sl=slice(start,min(start+256,len(ids)))
        atlas_delta[sl]=np.asarray(x[np.ix_(b,columns[sl])]).mean(0,dtype=np.float64)-np.asarray(x[np.ix_(a,columns[sl])]).mean(0,dtype=np.float64)
    reference_rank=residual_ranks(atlas_delta,strata)
    values=[]
    for cell in selected:
        info=cohort[cell['gsm']]
        path=HERE/('private/geo_early_schema_audit_01' if info['reused_pilot'] else 'private/geo_early_cohort_audit_01')/(cell['gsm']+'.HTSeq.output.txt.gz')
        if digest(path)!=info['file_sha256']:raise ValueError('Cell changed')
        with gzip.open(path,'rt') as f:counts={g:float(v) for g,v in (line.rstrip().split('\t') for line in f) if not g.startswith('__')}
        values.append(np.log1p(10000*np.array([counts[g] for g in ids])/sum(counts.values())))
    values=np.array(values)
    groups={(stage,region):np.array([i for i,r in enumerate(selected) if r['stage']==stage and r['broad_region']==region])
            for stage in ('E8.5','E9.5') for region in regions}
    if any(len(g)<2 for g in groups.values()):raise ValueError('Insufficient group for leave-one-out')
    def evaluate(indices):
        delta=np.mean([values[indices[('E9.5',r)]].mean(0)-values[indices[('E8.5',r)]].mean(0) for r in regions],axis=0)
        return correlation(residual_ranks(delta,strata),reference_rank)
    replay=evaluate(groups)
    expected=matched['results']['region_balanced']['stratified_rank_correlation']
    if abs(replay-expected)>1e-12:raise ValueError('Original matched diagnostic replay mismatch')
    bootstrap=[]
    for rep in range(64):
        seed=2026100101+rep;rng=np.random.default_rng(seed)
        indices={key:rng.choice(rows,size=len(rows),replace=True) for key,rows in groups.items()}
        row={'replicate':rep,'seed':seed,'stratified_rank_correlation':evaluate(indices),
             'sampled_gsm_ids':{str(k):[selected[i]['gsm'] for i in v] for k,v in indices.items()}}
        bootstrap.append(row);append_event(events,'bootstrap_replicate_completed',**row)
    loo=[]
    for i,cell in enumerate(selected):
        indices={k:v[v!=i] for k,v in groups.items()}
        row={'removed_gsm':cell['gsm'],'stage':cell['stage'],'broad_region':cell['broad_region'],
             'stratified_rank_correlation':evaluate(indices)}
        loo.append(row);append_event(events,'leave_one_out_completed',**row)
    results=np.array([r['stratified_rank_correlation'] for r in bootstrap]);lower=float(np.quantile(results,.025))
    minimum=min(r['stratified_rank_correlation'] for r in loo)
    passes=lower>0 and minimum>0
    report={'updated_utc':now(),'plan_sha256':digest(out/'plan.json'), 'original_replay_rho':replay,
        'bootstrap':bootstrap,'leave_one_out':loo,'bootstrap_mean_rho':float(results.mean()),
        'bootstrap_2_5_percentile_rho':lower,'bootstrap_97_5_percentile_rho':float(np.quantile(results,.975)),
        'bootstrap_nonpositive_replicates':int((results<=0).sum()),'leave_one_out_min_rho':minimum,
        'conditional_stability_gate_passed':passes,'independent_embryo_validation':False,
        'forecast_transfer_validated':False,'models_fit':0,'full_panel_scores':0,'reward_delta':0,
        'interpretation':'Selected-cell stability cannot establish cross-embryo or cross-assay transfer, forecast skill, or >72 readiness. Bootstrap and gene shuffles are diagnostics only.',
        'next_action':'Declare a leakage-safe source-only validation of external E8.5 static gene-program guidance; no E9.5-derived temporal coefficients in E8.5/E9.5 backtests.' if passes else 'Reject fragile external temporal prior; return to source-only saturating-time formula proxy.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    public=dict(report,report_sha256=digest(out/'report.json'))
    (HERE/'GSE76118_MATCHED_ROBUSTNESS_RESULTS.json').write_text(json.dumps(public,indent=2)+'\n')
    append_event(events,'robustness_completed',report_sha256=public['report_sha256'],gate_passed=passes)
    print(json.dumps({k:v for k,v in public.items() if k not in ('bootstrap','leave_one_out')}))


if __name__=='__main__':main()
