"""Validate public mouse growth-proxy markers on permitted historical rows."""
import json
from collections import Counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now,append_event


def fit_proxy_scores(x,rows,symbols,markers,seed=20260928):
    counts=Counter(symbols);lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
    all_markers=set().union(*[set(v) for v in markers.values()])
    means=np.zeros(x.shape[1])
    for a in range(0,x.shape[1],512):means[a:a+512]=np.asarray(x[np.ix_(rows,np.arange(a,min(a+512,x.shape[1])))]).mean(0,dtype=float)
    ranks=pd.Series(means).rank(method='min').to_numpy()-1
    bins=np.minimum((20*ranks/len(means)).astype(int),19)
    eligible=np.array([i for i,s in enumerate(symbols) if s in lookup and s not in all_markers])
    rng=np.random.default_rng(seed);scores={};audit={}
    for name,genes in markers.items():
        mapped=np.array([lookup[g] for g in genes if g in lookup]);controls=set()
        if len(mapped)<.8*len(genes):raise ValueError('Insufficient unique marker overlap: '+name)
        for g in mapped:
            pool=eligible[bins[eligible]==bins[g]]
            if len(pool)<20:raise ValueError('Insufficient expression-matched controls: '+name)
            controls.update(rng.choice(pool,20,replace=False).tolist())
        controls=np.array(sorted(controls))
        scores[name]=np.asarray(x[np.ix_(rows,mapped)],float).mean(1)-np.asarray(x[np.ix_(rows,controls)],float).mean(1)
        audit[name]={'requested_genes':len(genes),'unique_overlap':len(mapped),'control_genes':len(controls),
            'missing_or_ambiguous':sorted(set(genes)-set(lookup)),'marker_gene_indices':mapped.tolist(),'control_gene_indices':controls.tolist()}
    return scores,audit


def main():
    out=HERE/'private/growth_prior_audit_repair_01'
    if out.exists():raise ValueError('Preserve previous audit')
    data=HERE/'private/associated_prepared_01';reference=HERE/'private/growth_reference_01'
    ref=json.loads((HERE/'GROWTH_MARKER_REFERENCE.json').read_text())
    for record in ref['files']:
        if digest(reference/record['path'].split('/')[-1])!=record['sha256']:raise ValueError('Marker reference changed')
    prepared=json.loads((data/'report.json').read_text())
    for name,key in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(data/name)!=prepared[key]:raise ValueError('Prepared source changed')
    plan={'created_before_fit_utc':now(),'cutoffs':[8.,8.25],'seed':20260928,
        'source_sha256':digest(HERE/'growth_prior_audit.py'),'prepared_report_sha256':digest(data/'report.json'),
        'marker_reference_sha256':digest(HERE/'GROWTH_MARKER_REFERENCE.json'),
        'fit':'Permitted past rows only;20 mean-expression rank bins,20 control draws per marker, union of control genes excluding both marker sets. Controls include zero-expression genes so low-expression markers retain matched controls. Mean log-expression difference, not exact scanpy reproduction.',
        'repair':'Original growth_prior_audit_01 failed because its expressed-only control filter excluded low-expression-bin controls. Original plan, event and failure are preserved. Remove only that filter before any forecast scoring.',
        'limitations':'Mouse file named apoptosis is a P53-pathway proxy per author source. It is not a direct embryonic death measurement. Equal prepared per-stage sampling counts are not biological growth. No future expression or source annotations in learner.',
        'purpose':'Input/proxy audit only, not a scored forecast or validated birth/death rate.'}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2));(out/'growth_prior_audit.py').write_bytes((HERE/'growth_prior_audit.py').read_bytes())
    events=out/'events.jsonl';append_event(events,'growth_proxy_audit_plan_frozen',sha256=digest(out/'plan.json'))
    markers={'proliferation':sorted(set((reference/'mouse_proliferation.txt').read_text().splitlines())),
        'p53_proxy':sorted(set((reference/'mouse_apoptosis.txt').read_text().splitlines()))}
    symbols=pd.read_csv(data/'genes.csv').symbol.fillna('').tolist();meta=pd.read_csv(data/'selected_metadata.csv');stages=meta.numeric_stage.to_numpy(float)
    x=np.load(data/'expression.npy',mmap_mode='r');report={'plan':plan,'cutoffs':[],'status':'completed','forecast_score':None}
    for cutoff in plan['cutoffs']:
        rows=np.flatnonzero(stages<=cutoff);scores,audit=fit_proxy_scores(x,rows,symbols,markers)
        net=scores['proliferation']-scores['p53_proxy']
        np.savez_compressed(out/f'scores_{cutoff}.npz',rows=rows,proliferation=scores['proliferation'],p53_proxy=scores['p53_proxy'],net=net)
        result={'cutoff':cutoff,'past_rows':len(rows),'max_fit_stage':float(stages[rows].max()),'marker_audit':audit,
            'stage_summaries':[{'stage':float(t),'cells':int((stages[rows]==t).sum()),'proliferation_mean':float(scores['proliferation'][stages[rows]==t].mean()),'p53_proxy_mean':float(scores['p53_proxy'][stages[rows]==t].mean()),'net_quantiles':np.quantile(net[stages[rows]==t],[.05,.5,.95]).tolist()} for t in np.unique(stages[rows])],
            'scores_sha256':digest(out/f'scores_{cutoff}.npz')}
        report['cutoffs'].append(result);append_event(events,'growth_proxy_scores_fitted',cutoff=cutoff,past_rows=len(rows))
    (out/'report.json').write_text(json.dumps(report,indent=2));(HERE/'GROWTH_PRIOR_AUDIT.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'status':report['status'],'overlap':{r['cutoff']:{k:v['unique_overlap'] for k,v in r['marker_audit'].items()} for r in report['cutoffs']}}))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
