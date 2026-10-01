"""Capture-balanced log-gene-mean hindcast; diagnostic, not benchmark scoring."""
import json
from collections import Counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now,append_event
from lineage_residual_screen import lineage
from projection_survival_diagnostics import PastReadGuard
from capture_mean_shrinkage import forecast
from test_capture_mean_shrinkage import main as controls

RUN='capture_mean_shrinkage_audit_01'
PUBLIC='CAPTURE_MEAN_SHRINKAGE_AUDIT_RESULTS.json'


def main():
    out=HERE/'private'/RUN
    if out.exists():raise ValueError('Frozen run exists; never duplicate')
    spec_path=HERE/'NEXT_CAPTURE_MEAN_SHRINKAGE_AUDIT.json'
    spec=json.loads(spec_path.read_text());source=HERE/'private/associated_prepared_01'
    prepared=json.loads((source/'report.json').read_text())
    for name,key in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(source/name)!=prepared[key]:raise ValueError('Source hash changed')
    m=pd.read_csv(source/'selected_metadata.csv');stages=m.numeric_stage.to_numpy(float)
    captures=m['sample'].astype(str).to_numpy();labels=m.celltype_extended_atlas.map(lineage).to_numpy()
    symbols=pd.read_csv(source/'genes.csv').symbol.fillna('').tolist();counts=Counter(symbols)
    panel_path=HERE.parents[1]/'outputs/t1_run/T1__val.genes.txt';panel=set(panel_path.read_text().splitlines())
    columns=np.array([i for i,s in enumerate(symbols) if s in panel and counts[s]==1])
    x=np.load(source/'expression.npy',mmap_mode='r')
    plan={**spec,'created_utc':now(),'spec_sha256':digest(spec_path),'prepared_report_sha256':digest(source/'report.json'),
          'panel_sha256':digest(panel_path),'mapped_unique_genes':len(columns),
          'code_sha256':{n:digest(HERE/n) for n in ['capture_mean_shrinkage_audit.py','capture_mean_shrinkage.py','test_capture_mean_shrinkage.py','projection_survival_diagnostics.py','lineage_residual_screen.py']}}
    out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2));(out/'executed_source.py').write_bytes((HERE/'capture_mean_shrinkage_audit.py').read_bytes())
    events=out/'events.jsonl';append_event(events,'plan_frozen',sha256=digest(out/'plan.json'));controls()
    def mean(rows,reader):
        result=np.zeros(len(columns))
        for start in range(0,len(columns),256):
            col=columns[start:start+256];result[start:start+len(col)]=np.asarray(reader[np.ix_(rows,col)],float).mean(0)
        return result
    frozen=[]
    # All three historical forecasts are frozen before any corresponding target read.
    for cutoff,target in spec['folds']:
        guard=PastReadGuard(x,stages,cutoff);previous=cutoff-.25
        folder=out/f'cutoff_{cutoff}';folder.mkdir();records=[];arrays={}
        for group in spec['lineages']:
            stage_means=[];stage_cells=[];support=[]
            for t in [previous,cutoff]:
                qualifying=[];rows_all=[];entries=[]
                for cap in sorted(set(captures[stages==t])):
                    rows=np.flatnonzero((stages==t)&(labels==group)&(captures==cap))
                    if len(rows)>=spec['min_capture_cells']:
                        qualifying.append(mean(rows,guard));rows_all.extend(rows.tolist());entries.append({'capture':cap,'cells':len(rows)})
                stage_means.append(np.stack(qualifying) if qualifying else None)
                stage_cells.append(np.array(rows_all,dtype=int));support.append(entries)
            if min(len(s) for s in support)<2:
                records.append({'lineage':group,'status':'unsupported','support':support,'reason':'Fewer than two eligible captures at a fit stage; no calibrated shrinkage forecast'})
                continue
            candidates,audit=forecast(*stage_means)
            ca,cb=[mean(rows,guard) for rows in stage_cells]
            candidates['cell_weighted_linear']=cb+(cb-ca)
            for name,value in candidates.items():arrays[group+'__'+name]=value
            records.append({'lineage':group,'status':'supported','support':support,'fit_audit':audit})
        np.savez(folder/'means.npz',**arrays,columns=columns)
        append_event(events,'past_forecasts_frozen',cutoff=cutoff,target=target,max_fit_stage=guard.max_stage,sha256=digest(folder/'means.npz'))
        frozen.append({'cutoff':cutoff,'target':target,'records':records,'forecast_sha256':digest(folder/'means.npz')})
    append_event(events,'all_forecasts_frozen_before_evaluation')
    results=[]
    for case in frozen:
        with np.load(out/f"cutoff_{case['cutoff']}"/'means.npz') as predictions:
            for record in case['records']:
                if record['status']!='supported':continue
                group=record['lineage'];base=predictions[group+'__persistence']
                for cap in sorted(set(captures[stages==case['target']])):
                    rows=np.flatnonzero((stages==case['target'])&(labels==group)&(captures==cap))
                    if len(rows)<spec['min_capture_cells']:continue
                    observed=mean(rows,x);base_error=float(np.mean((base-observed)**2))
                    diagnostics=[]
                    for name in spec['candidates']:
                        error=float(np.mean((predictions[group+'__'+name]-observed)**2))
                        diagnostics.append({'candidate':name,'log_gene_mean_mse':error,'difference_vs_persistence':error-base_error})
                    results.append({'cutoff':case['cutoff'],'target':case['target'],'lineage':group,'held_capture':cap,'cells':len(rows),'diagnostics':diagnostics})
    report={'status':'completed','plan':plan,'fit_cases':frozen,'results':results,'headline_score':None,'raw_metrics':None,'skills':None,'reward_delta':0,'full_panel_scoring_executed':False,
            'summary':[{'candidate':name,'capture_lineage_cases':len(results),'better_than_persistence':sum(next(v for v in r['diagnostics'] if v['candidate']==name)['difference_vs_persistence']<0 for r in results)} for name in spec['candidates']]}
    (out/'report.json').write_text(json.dumps(report,indent=2));(HERE/PUBLIC).write_text(json.dumps({**report,'report_sha256':digest(out/'report.json')},indent=2))
    append_event(events,'audit_completed',report_sha256=digest(out/'report.json'),reward_delta=0)
    print(json.dumps({'status':'completed','summary':report['summary'],'report_sha256':digest(out/'report.json')}))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
