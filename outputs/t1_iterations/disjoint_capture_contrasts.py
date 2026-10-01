"""Past-only split-middle capture contrast diagnostic, not independent embryo evidence."""
import json
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now,append_event
from lineage_residual_screen import lineage


def rho(a,b):
    if np.std(a)<1e-12 or np.std(b)<1e-12:return None
    return float(spearmanr(a,b).statistic)


def main():
    source=HERE/'private/associated_prepared_01';out=HERE/'private/disjoint_capture_contrasts_01'
    if out.exists():raise ValueError('Preserve original diagnostic')
    metadata=pd.read_csv(source/'selected_metadata.csv');metadata['lineage']=metadata.celltype_extended_atlas.map(lineage)
    integrity_path=HERE/'PAST_SOURCE_INTEGRITY_RESULTS.json';integrity=json.loads(integrity_path.read_text())
    groups=integrity['common_supported_lineages'];triples=[(7.5,7.75,8.),(7.75,8.,8.25)]
    partitions=[]
    for a,b,c in triples:
        captures=sorted(int(v) for v in metadata.loc[metadata.numeric_stage==b,'sample'].unique())
        if len(captures)!=4:raise ValueError('Expected four middle-stage captures')
        for left in combinations(captures,2):partitions.append({'a':a,'b':b,'c':c,'left':list(left),'right':[v for v in captures if v not in left]})
    plan={'created_utc':now(),'code_sha256':digest(HERE/'disjoint_capture_contrasts.py'),'integrity_report_sha256':digest(integrity_path),
          'source_report_sha256':digest(source/'report.json'),'metadata_sha256':digest(source/'selected_metadata.csv'),
          'groups':groups,'minimum_cells_per_group':20,'partitions':partitions,
          'primary':'All seven predeclared lineages must have >=20 cells in both middle partitions and both endpoints. Otherwise primary unavailable; secondary common-supported-subset diagnosis explicitly labeled.',
          'scope':'Past-only <=8.25 expression. Twelve oriented contrasts (six per triple, three complementary partitions); no independent embryo or64replicate claim, fit, future expression, prediction, benchmark score or reward.'}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');(out/'executed_source.py').write_bytes((HERE/'disjoint_capture_contrasts.py').read_bytes())
    events=out/'events.jsonl';append_event(events,'plan_frozen',sha256=digest(out/'plan.json'))
    manifest=json.loads((source/'report.json').read_text())
    if digest(source/'expression.npy')!=manifest['expression_sha256'] or plan['metadata_sha256']!=manifest['metadata_sha256']:raise ValueError('Source changed')
    x=np.load(source/'expression.npy',mmap_mode='r');past=metadata.loc[metadata.numeric_stage<=8.25];group_stats={}
    for (stage,sample,group),part in past.groupby(['numeric_stage','sample','lineage']):
        if group not in groups:continue
        ids=part.index.to_numpy();total=np.zeros(x.shape[1])
        for start in range(0,len(ids),64):total+=np.asarray(x[ids[start:start+64]],dtype=np.float64).sum(0)
        group_stats[(float(stage),int(sample),group)]=(total,len(ids))
    def grouped(stage,samples,group):
        selected=[v for (st,sa,g),v in group_stats.items() if st==stage and (samples is None or sa in samples) and g==group]
        count=sum(v[1] for v in selected)
        return (sum(v[0] for v in selected)/count,count) if count else (None,0)
    rows=[]
    for partition in partitions:
        a,b,c=partition['a'],partition['b'],partition['c'];vectors={};coverage={};eligible=[]
        for g in groups:
            stats=[grouped(a,None,g),grouped(b,None,g),grouped(c,None,g),grouped(b,partition['left'],g),grouped(b,partition['right'],g)]
            coverage[g]=dict(zip(['a','b_full','c','b_left','b_right'],[v[1] for v in stats]))
            if min(v[1] for v in stats)<20:continue
            eligible.append(g);vectors[g]=[v[0] for v in stats]
        row={**partition,'cell_overlap_between_middle_partitions':0,'coverage':coverage,'supported_groups':eligible,
             'primary_available':len(eligible)==len(groups),'primary_unavailable_reason':None if len(eligible)==len(groups) else 'Predeclared lineage coverage <20cells'}
        if eligible:
            A,B,C,L,R=[np.mean([vectors[g][i] for g in eligible],axis=0) for i in range(5)]
            d1=B-A;d2=C-B;split1=L-A;split2=C-R
            per_group={g:{'shared_middle_rho':rho(v[1]-v[0],v[2]-v[1]),'disjoint_middle_rho':rho(v[3]-v[0],v[2]-v[4])} for g,v in vectors.items()}
            row['diagnostics']={'scope':'primary_all_predeclared_lineages' if row['primary_available'] else 'secondary_supported_subset_only',
               'shared_middle_rho':rho(d1,d2),'disjoint_middle_rho':rho(split1,split2),
               'left_vs_right_middle_mean_rho':rho(L,R),'middle_difference_std':float(np.std(L-R)),
               'middle_difference_vs_shared_delta_std_ratio':float(np.std(L-R)/max((np.std(d1)+np.std(d2))/2,1e-12)),
               'disjoint_delta_sign_agreement':float((np.sign(split1)==np.sign(split2)).mean()),
               'disjoint_vs_shared_first_delta_rho':rho(split1,d1),'disjoint_vs_shared_second_delta_rho':rho(split2,d2),'per_group':per_group}
        rows.append(row);append_event(events,'partition_diagnosed',**row)
    summaries=[]
    for a,b,c in triples:
        selected=[v for v in rows if v['a']==a and v['b']==b and v['c']==c];primary=[v for v in selected if v['primary_available']]
        summaries.append({'triple':[a,b,c],'oriented_partitions':len(selected),'unique_complementary_partitions':3,'primary_evaluable':len(primary),
            'primary_shared_middle_rho':primary[0]['diagnostics']['shared_middle_rho'] if primary else None,
            'primary_disjoint_middle_rho_range':[min(v['diagnostics']['disjoint_middle_rho'] for v in primary),max(v['diagnostics']['disjoint_middle_rho'] for v in primary)] if primary else None,
            'primary_disjoint_middle_rho_median':float(np.median([v['diagnostics']['disjoint_middle_rho'] for v in primary])) if primary else None})
    report={'updated_utc':now(),'plan_sha256':digest(out/'plan.json'),'summary':summaries,'partitions':rows,'scope':plan['scope'],
            'embryo_id_status':integrity['embryo_id_status'],'causal_discontinuity_established':False,'new_scores':0,'reward_delta':0}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');report['report_sha256']=digest(out/'report.json')
    (HERE/'DISJOINT_CAPTURE_CONTRAST_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'completed':True,'summary':summaries,'report_sha256':report['report_sha256']}))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
