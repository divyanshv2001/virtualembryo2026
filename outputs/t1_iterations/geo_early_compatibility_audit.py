"""Cached external/atlas assay compatibility diagnostic; no forecast or score."""
import gzip
import json
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event


def group(label):
    if label.startswith('Cardiomyocytes'): return 'cardiomyocyte'
    if 'cardiopharyngeal' in label.lower(): return 'cardiopharyngeal_progenitor'
    if label == 'Endocardium': return 'endocardium'
    if label == 'Epicardium': return 'epicardium'
    if label == 'Erythroid': return 'erythroid'
    return 'other_associated'


def main():
    out=HERE/'private/geo_early_compatibility_audit_01'
    if out.exists(): raise ValueError('Preserve prior compatibility diagnostic')
    source=HERE/'private/associated_prepared_01'
    manifest=json.loads((source/'report.json').read_text())
    for name,key in [('expression.npy','expression_sha256'),('genes.csv','genes_sha256'),('selected_metadata.csv','metadata_sha256')]:
        if digest(source/name)!=manifest[key]: raise ValueError('Atlas input changed')
    mapping_path=HERE/'private/geo_early_prospective_policy_01/stable_id_panel_mapping.json'
    cohort_path=HERE/'GSE76118_EARLY_COHORT_RESULTS.json'
    cohort=json.loads(cohort_path.read_text())
    mapping=json.loads(mapping_path.read_text())
    genes=pd.read_csv(source/'genes.csv').fillna('')
    frequency=Counter(genes.symbol)
    atlas_lookup={r.symbol:i for i,r in enumerate(genes.itertuples()) if r.symbol and frequency[r.symbol]==1}
    sample_genes=set()
    # All schema files contain the same frozen identifier panel; use a cached pilot.
    with gzip.open(HERE/'private/geo_early_schema_audit_01/GSM2035425.HTSeq.output.txt.gz','rt') as stream:
        sample_genes={line.split('\t')[0] for line in stream if not line.startswith('__')}
    ids=sorted(g for g,s in mapping.items() if g in sample_genes and s in atlas_lookup)
    columns=np.array([atlas_lookup[mapping[g]] for g in ids])
    meta=pd.read_csv(source/'selected_metadata.csv')
    labels=meta.celltype_extended_atlas.map(group).to_numpy()
    stages=meta.numeric_stage.to_numpy(float)
    reference=np.flatnonzero(stages==8.5)
    groups=sorted(set(labels[reference]))
    plan={'created_utc':now(),'code_sha256':digest(HERE/'geo_early_compatibility_audit.py'),
        'atlas_report_sha256':digest(source/'report.json'),'cohort_sha256':digest(cohort_path),
        'mapping_sha256':digest(mapping_path),'diagnostic_reference_stage':8.5,
        'reference_rows':len(reference),'shared_unique_genes':len(ids),'groups':groups,
        'features':'Top64 group-discriminative genes by between-centroid variance / within-group variance +1e-4, from E8.5 atlas only; stable index ties.',
        'comparison':'Group centroid cosine after E8.5 atlas centering/scaling; diagnostic similarity, not validated cell identity. Log1p(10000*assignedcount/totalassigned) for external; atlas same numeric normalization target.',
        'scope':'E8.5 and E9.5 cohort assay/lineage compatibility only; E9.5 never fits E8.5 diagnostic reference. No transfer forecast/full-panel metric/reward. Neither sample identity nor label ensures independent embryo replication.',
        'resources':'D cached files and atlas memmap;256gene chunks; no large generated matrix.'}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (out/'executed_source.py').write_bytes((HERE/'geo_early_compatibility_audit.py').read_bytes())
    events=out/'events.jsonl';append_event(events,'plan_frozen',plan_sha256=digest(out/'plan.json'))
    x=np.load(source/'expression.npy',mmap_mode='r')
    means=np.empty((len(groups),len(ids)));within=np.zeros(len(ids));center=np.zeros(len(ids));scale=np.zeros(len(ids))
    reference_detected=np.zeros(len(reference))
    for start in range(0,len(ids),256):
        sl=slice(start,min(start+256,len(ids)))
        values=np.asarray(x[np.ix_(reference,columns[sl])],dtype=np.float32)
        center[sl]=values.mean(0);scale[sl]=np.maximum(values.std(0),.05)
        reference_detected+=(values>0).sum(1)
        for i,g in enumerate(groups):
            mask=labels[reference]==g;means[i,sl]=values[mask].mean(0)
            within[sl]+=values[mask].var(0)*mask.mean()
    feature_score=means.var(0)/(within+1e-4)
    features=np.argsort(-feature_score,kind='stable')[:64]
    centroids=(means[:,features]-center[features])/scale[features]
    centroids/=np.maximum(np.linalg.norm(centroids,axis=1,keepdims=True),1e-8)
    np.savez_compressed(out/'frozen_diagnostic.npz',features=features,columns=columns[features],
                        gene_ids=np.array(ids)[features],groups=np.array(groups),center=center[features],scale=scale[features],centroids=centroids)
    append_event(events,'reference_frozen_before_external_read',sha256=digest(out/'frozen_diagnostic.npz'))
    rows=[];stage_values={8.5:[],9.5:[]}
    for sample in cohort['outcomes']:
        if not sample.get('prospective_qc_pass'): continue
        path=HERE/'private/geo_early_cohort_audit_01'/(sample['gsm']+'.HTSeq.output.txt.gz')
        if sample['reused_pilot']:path=HERE/'private/geo_early_schema_audit_01'/path.name
        if digest(path)!=sample['file_sha256']:raise ValueError('External cell changed')
        with gzip.open(path,'rt') as stream:
            counts={g:float(v) for g,v in (line.rstrip().split('\t') for line in stream) if not g.startswith('__')}
        values=np.log1p(10000*np.array([counts[g] for g in ids])/sum(counts.values()))
        z=(values[features]-center[features])/scale[features]
        cosine=centroids@z/max(np.linalg.norm(z),1e-8)
        order=np.argsort(-cosine,kind='stable')
        stage=float(sample['stage'][1:]);stage_values[stage].append(values)
        rows.append({'gsm':sample['gsm'],'stage':sample['stage'],'region':sample['region'],
            'diagnostic_nearest_group':groups[order[0]],'cosine':dict(zip(groups,cosine.tolist())),
            'top_two_similarity_margin':float(cosine[order[0]]-cosine[order[1]]),
            'shared_gene_detection_fraction':float((values>0).mean()),
            'atlas_reference_detection_percentile':float((reference_detected/len(ids)<(values>0).mean()).mean()),
            'cell_identity_validated':False})
    external_delta=np.mean(stage_values[9.5],axis=0)-np.mean(stage_values[8.5],axis=0)
    atlas_delta=np.empty(len(ids));cardiac=np.isin(labels,['cardiomyocyte','cardiopharyngeal_progenitor','endocardium','epicardium'])
    counts_by_stage={}
    for stage in (8.5,9.5):counts_by_stage[str(stage)]=int(((stages==stage)&cardiac).sum())
    for start in range(0,len(ids),256):
        sl=slice(start,min(start+256,len(ids)))
        a=np.flatnonzero((stages==8.5)&cardiac);b=np.flatnonzero((stages==9.5)&cardiac)
        atlas_delta[sl]=np.asarray(x[np.ix_(b,columns[sl])]).mean(0)-np.asarray(x[np.ix_(a,columns[sl])]).mean(0)
    report={'updated_utc':now(),'plan_sha256':digest(out/'plan.json'),'outcomes':rows,
        'shared_unique_genes':len(ids),'diagnostic_group_counts':dict(Counter(r['diagnostic_nearest_group'] for r in rows)),
        'external_detection_fraction_range':[min(r['shared_gene_detection_fraction'] for r in rows),max(r['shared_gene_detection_fraction'] for r in rows)],
        'atlas_E8_5_detection_fraction_quantiles':np.quantile(reference_detected/len(ids),[.05,.5,.95]).tolist(),
        'pooled_external_vs_atlas_cardiac_temporal_delta_spearman':float(spearmanr(external_delta,atlas_delta).statistic),
        'atlas_cardiac_counts_by_stage':counts_by_stage,'models_fit':0,'full_panel_scores':0,'reward_delta':0,
        'transfer_validated':False,'confounding':'External stage-region composition and cell types are not matched; pooled delta correlation is diagnostic only. Full-length read counts and atlas UMI assay may have different detection/length biases. Capture ID is not independent embryo ID.',
        'next_gate':'Use these diagnostics to declare a narrow matched-lineage, magnitude-normalized external temporal direction prior with zero/shuffled-direction controls, or reject transfer if overlap/detection incompatible. No direct pooled extrapolation.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    public=dict(report,report_sha256=digest(out/'report.json'))
    (HERE/'GSE76118_EARLY_COMPATIBILITY_RESULTS.json').write_text(json.dumps(public,indent=2)+'\n')
    append_event(events,'compatibility_audit_completed',report_sha256=public['report_sha256'])
    print(json.dumps({k:v for k,v in public.items() if k!='outcomes'}))


if __name__=='__main__':main()
