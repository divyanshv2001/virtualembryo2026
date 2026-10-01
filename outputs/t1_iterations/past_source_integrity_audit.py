"""Past-only source stage/capture integrity diagnostics, no forecasting."""
import json
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now,append_event
from lineage_residual_screen import lineage


def quant(x):return {str(q):float(np.quantile(x,q)) for q in [.05,.5,.95]}
def rho(a,b):return float(spearmanr(a,b).statistic)


def main():
    out=HERE/'private/past_source_integrity_01'
    if out.exists():raise ValueError('Preserve audit evidence')
    source=HERE/'private/associated_prepared_01';manifest=json.loads((source/'report.json').read_text())
    plan={'created_utc':now(),'code_sha256':digest(HERE/'past_source_integrity_audit.py'),
          'source_report_sha256':digest(source/'report.json'),'raw_library_sha256':digest(source/'raw_library_sizes.npy'),
          'stages':[7.5,7.75,8.,8.25],'minimum_group_cells':20,'scope':'Past-only source expression/stage/capture diagnostics. No future expression, fitting, predictions, benchmark scores or reward. Capture sample labels are not established embryo IDs.',
          'dependencies_sha256':{'lineage_residual_screen.py':digest(HERE/'lineage_residual_screen.py')}}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');(out/'executed_source.py').write_bytes((HERE/'past_source_integrity_audit.py').read_bytes())
    events=out/'events.jsonl';append_event(events,'plan_frozen',sha256=digest(out/'plan.json'))
    for f,k in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(source/f)!=manifest[k]:raise ValueError('Source changed: '+f)
    x=np.load(source/'expression.npy',mmap_mode='r');libs=np.load(source/'raw_library_sizes.npy',mmap_mode='r')
    m=pd.read_csv(source/'selected_metadata.csv');m['lineage']=m.celltype_extended_atlas.map(lineage)
    if len(libs)!=len(m):raise ValueError('Library row mismatch')
    stats={};aggregates={};stage_rows=[];sample_rows=[]
    def aggregate(rows):
        sums=np.zeros(x.shape[1]);det=np.zeros(x.shape[1]);cell_det=[];mass=[]
        for start in range(0,len(rows),64):
            block=np.asarray(x[rows[start:start+64]],dtype=np.float64)
            sums+=block.sum(0);det+=(block>0).sum(0);cell_det.extend((block>0).mean(1));mass.extend(np.expm1(block).sum(1))
        return sums/len(rows),det/len(rows),np.asarray(cell_det),np.asarray(mass)
    for stage in plan['stages']:
        rows=m.index[m.numeric_stage==stage].to_numpy();mean,det,cell_det,mass=aggregate(rows)
        stats[stage]=(mean,det);local=m.loc[rows]
        rec={'stage':stage,'cells':len(rows),'sample_counts':{str(k):int(v) for k,v in local['sample'].value_counts().items()},
             'version_counts':{str(k):int(v) for k,v in local.embryo_version.value_counts().items()},
             'lineage_fractions':{str(k):float(v) for k,v in local.lineage.value_counts(normalize=True).items()},
             'raw_library_quantiles':quant(libs[rows]),'cell_detection_fraction_quantiles':quant(cell_det),
             'normalized_implied_count_quantiles':quant(mass),'max_normalization_relative_error':float(np.max(np.abs(mass/manifest['normalization_target']-1)))}
        stage_rows.append(rec);append_event(events,'stage_audited',**rec)
        for (sample,group),part in local.groupby(['sample','lineage']):
            ids=part.index.to_numpy();gm,gd,_,_=aggregate(ids);aggregates[(stage,int(sample),group)]=(gm,gd,len(ids))
        for sample,part in local.groupby('sample'):
            ids=part.index.to_numpy();sm,sd,sdet,smass=aggregate(ids)
            sample_rows.append({'stage':stage,'sample':int(sample),'cells':len(ids),'raw_library_quantiles':quant(libs[ids]),'cell_detection_quantiles':quant(sdet),
                'lineage_fractions':{str(k):float(v) for k,v in part.lineage.value_counts(normalize=True).items()},'mean_vs_stage_rho':rho(sm,mean)})
    group_means={};group_counts={}
    for stage in plan['stages']:
        for group in sorted(m.loc[m.numeric_stage==stage,'lineage'].unique()):
            entries=[v for (st,sa,g),v in aggregates.items() if st==stage and g==group];total=sum(v[2] for v in entries)
            if total>=plan['minimum_group_cells']:
                group_means[(stage,group)]=sum(v[0]*v[2] for v in entries)/total;group_counts[(stage,group)]=total
    common=sorted(set.intersection(*[{g for st,g in group_means if st==stage} for stage in plan['stages']]))
    standardized={stage:np.mean([group_means[(stage,g)] for g in common],axis=0) for stage in plan['stages']}
    comparisons=[];deltas={}
    for a,b in zip(plan['stages'][:-1],plan['stages'][1:]):
        bulk=stats[b][0]-stats[a][0];standard=standardized[b]-standardized[a];deltas[b]=(bulk,standard)
        comparisons.append({'start':a,'end':b,'bulk_vs_equal_lineage_delta_rho':rho(bulk,standard),'bulk_delta_std':float(np.std(bulk)),
            'equal_lineage_delta_std':float(np.std(standard)),'per_lineage_bulk_delta_rho':{g:rho(bulk,group_means[(b,g)]-group_means[(a,g)]) for g in common}})
    stability=[]
    for a,b in [(7.75,8.),(8.,8.25)]:
        row={'earlier_delta_ending':a,'later_delta_ending':b,'bulk_delta_rho':rho(deltas[a][0],deltas[b][0]),
             'equal_lineage_delta_rho':rho(deltas[a][1],deltas[b][1]),
             'per_lineage_consecutive_delta_rho':{g:rho(group_means[(a,g)]-group_means[(a-.25,g)],group_means[(b,g)]-group_means[(b-.25,g)]) for g in common}}
        stability.append(row)
    leaveout=[]
    for a,b in zip(plan['stages'][:-1],plan['stages'][1:]):
        full=standardized[b]-standardized[a]
        for st in [a,b]:
            for sample in sorted(m.loc[m.numeric_stage==st,'sample'].unique()):
                remaining=[]
                for g in common:
                    entries=[v for (s,sa,group),v in aggregates.items() if s==st and sa!=sample and group==g];total=sum(v[2] for v in entries)
                    if total<plan['minimum_group_cells']:break
                    remaining.append(sum(v[0]*v[2] for v in entries)/total)
                if len(remaining)!=len(common):
                    leaveout.append({'start':a,'end':b,'removed_stage':st,'removed_sample':int(sample),'status':'insufficient_lineage_coverage'});continue
                changed=np.mean(remaining,axis=0);delta=standardized[b]-changed if st==a else changed-standardized[a]
                mask=np.abs(full)>1e-6
                leaveout.append({'start':a,'end':b,'removed_stage':st,'removed_sample':int(sample),'status':'diagnosed','rho_vs_complete':rho(delta,full),'material_sign_agreement':float((np.sign(delta[mask])==np.sign(full[mask])).mean())})
    np.savez_compressed(out/'past_stage_summary_vectors.npz',**{f'mean_{st}':v[0] for st,v in stats.items()},**{f'equal_lineage_{st}':v for st,v in standardized.items()})
    report={'updated_utc':now(),'plan_sha256':digest(out/'plan.json'),'scope':plan['scope'],'embryo_id_status':manifest['embryo_id_status'],
            'sample_ids_shared_between_stages':len(set.intersection(*[set(m.loc[m.numeric_stage==st,'sample']) for st in plan['stages']])),
            'stages':stage_rows,'samples':sample_rows,'common_supported_lineages':common,'transitions':comparisons,'consecutive_delta_stability':stability,
            'leave_one_capture_out':leaveout,'summary_vectors_sha256':digest(out/'past_stage_summary_vectors.npz'),
            'normalization_target':manifest['normalization_target'],'causal_discontinuity_established':False,'new_scores':0,'reward_delta':0}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');report['report_sha256']=digest(out/'report.json')
    (HERE/'PAST_SOURCE_INTEGRITY_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'completed':True,'report_sha256':report['report_sha256']}))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
