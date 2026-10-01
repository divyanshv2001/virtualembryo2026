"""Past-only capture median and matched cell-weighted lineage temporal contrasts."""
import numpy as np
from scipy.stats import spearmanr


def fit_capture_trends(x,metadata,columns,cutoff,min_cells=10,min_captures=2):
    times=[cutoff-.5,cutoff-.25,cutoff];stats={};fallbacks=[]
    past=metadata.loc[metadata.numeric_stage.isin(times)]
    for (stage,sample,group),part in past.groupby(['numeric_stage','sample','lineage']):
        ids=part.index.to_numpy();sums=np.zeros(len(columns))
        for start in range(0,len(ids),64):sums+=np.asarray(x[np.ix_(ids[start:start+64],columns)],dtype=np.float64).sum(0)
        stats[(float(stage),int(sample),group)]=(sums,len(ids))
    groups=sorted(past.lineage.unique());weights=metadata.loc[metadata.numeric_stage==cutoff,'lineage'].value_counts(normalize=True).to_dict()
    slope_weights=np.array([-.5,0.,.5])  # equal-stage OLS; one quarterday step
    def compute(omit=None):
        cell=np.zeros(len(columns));robust=np.zeros(len(columns));coverage=[]
        for group in groups:
            cell_means=[];robust_means=[];supported=True
            for stage in times:
                entries=[v for (st,sa,g),v in stats.items() if st==stage and g==group and (omit is None or (st,sa)!=omit)]
                total=sum(v[1] for v in entries)
                if total==0:
                    supported=False;coverage.append({'stage':stage,'lineage':group,'status':'missing_group_zero_slope','capture_count':0});break
                mean=sum(v[0] for v in entries)/total
                eligible=[v[0]/v[1] for v in entries if v[1]>=min_cells]
                use_median=len(eligible)>=min_captures
                median=np.median(np.stack(eligible),axis=0) if use_median else mean
                cell_means.append(mean);robust_means.append(median)
                coverage.append({'stage':stage,'lineage':group,'cells':total,'eligible_captures':len(eligible),'status':'capture_median' if use_median else 'pooled_fallback'})
            if supported:
                w=weights.get(group,0.)
                cell+=w*(slope_weights@np.stack(cell_means));robust+=w*(slope_weights@np.stack(robust_means))
        return cell,robust,coverage
    cell,robust,coverage=compute();jackknife=[]
    for stage in times:
        captures=sorted(int(v) for v in past.loc[past.numeric_stage==stage,'sample'].unique())
        for sample in captures:
            c,r,cov=compute((stage,sample))
            jackknife.append({'held_out_stage':stage,'held_out_capture':sample,'scope':'past-fit sensitivity only, not independent/temporal forecast validation',
                'median_slope_rho_vs_full':float(spearmanr(r,robust).statistic),'cell_slope_rho_vs_full':float(spearmanr(c,cell).statistic),
                'pooled_fallback_entries':sum(v['status']=='pooled_fallback' for v in cov),'missing_entries':sum(v['status']=='missing_group_zero_slope' for v in cov)})
    return {'slope_cellweighted':cell,'slope_capturemedian':robust},{'past_stages':times,'current_lineage_fractions':weights,'coverage':coverage,'leave_one_capture_out':jackknife,
        'formula':'Each lineage has equal-stage three-point OLS; capture-median means compared with matched pooled cell means. Both aggregate with identical current lineage fractions. No future input or target-based selection.'}
