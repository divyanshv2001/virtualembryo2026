"""Post-score diagnostics; never feeds held-out quantities back into a learner."""
import json
from collections import Counter
import numpy as np
import pandas as pd
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now


def main():
    folder=HERE/'private/neural_ode_horizon_01'
    report_path=folder/'report.json';report=json.loads(report_path.read_text())
    if report['status']!='completed':raise ValueError('Require completed frozen forecasts and scoring')
    data=HERE/'private/associated_prepared_01'
    if digest(data/'report.json')!=report['plan']['prepared_report_sha256']:raise ValueError('Prepared data changed')
    prepared=json.loads((data/'report.json').read_text())
    for name,key in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(data/name)!=prepared[key]:raise ValueError('Prepared source changed')
    root=HERE.parents[1];panel=(root/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
    symbols=pd.read_csv(data/'genes.csv').symbol.fillna('').tolist();counts=Counter(symbols)
    lookup={s:i for i,s in enumerate(symbols) if s and counts[s]==1}
    mapped=np.array([i for i,s in enumerate(panel) if s in lookup]);atlas=np.array([lookup[panel[i]] for i in mapped])
    x=np.load(data/'expression.npy',mmap_mode='r');stages=pd.read_csv(data/'selected_metadata.csv').numeric_stage.to_numpy(float)
    output={'created_utc':now(),'report_sha256':digest(report_path),'scope':'Post-score descriptive diagnostics on already used development folds. These are not DE hypothesis-test scores, independent validation or learner inputs. Correlations use held-out means only after forecasts were frozen.',
        'interpretation':'The audited DE score uses the truth effect threshold to define the number of up/down genes, then ranks predicted gene changes. Predicted changes need not exceed.25. Increasing magnitude alone does not correct a wrong predicted ranking; these descriptive counts cannot explain DE-score failure by themselves. Correlations here are ordinary mapped-gene correlations, not the official full-panel rank-partial direction metric. Diagnostics average in float64; original scores retain scorer source precision.', 'folds':[]}
    for fold in report['folds']:
        f=folder/f"cutoff_{fold['cutoff']}";donors=np.load(f/'copy.npy',mmap_mode='r')
        if digest(f/'copy.npy')!=fold['generation']['copy']['prediction_sha256']:raise ValueError('Donor control changed')
        donor_mean=donors.mean(0,dtype=float)
        target_rows=np.sort(np.random.default_rng(report['plan']['seed']).choice(np.flatnonzero(stages==fold['target']),2000,replace=False))
        order=np.random.default_rng(report['plan']['seed']).permutation(2000)
        truth=np.asarray(x[np.ix_(target_rows[order[:1000]],atlas)],float)
        target_delta=truth.mean(0)-donor_mean[mapped]
        meaningful=np.abs(target_delta)>=.25
        record={'cutoff':fold['cutoff'],'target':fold['target'],'mapped_genes':len(mapped),'target_genes_with_abs_mean_change_ge_point25':int(meaningful.sum()),'candidates':[]}
        for result in fold['results']:
            name=result['candidate'];path=f/(name+'.npy')
            if digest(path)!=fold['generation'][name]['prediction_sha256']:raise ValueError('Forecast changed')
            pred=np.load(path,mmap_mode='r');delta=pred.mean(0,dtype=float)[mapped]-donor_mean[mapped]
            corr=float(np.corrcoef(delta,target_delta)[0,1]) if delta.std()>0 and target_delta.std()>0 else None
            record['candidates'].append({'candidate':name,'local_score':result['local_score'],
                'mean_abs_gene_change':float(np.abs(delta).mean()),'absolute_gene_change_quantiles':np.quantile(np.abs(delta),[.5,.9,.99,1.]).tolist(),
                'genes_with_abs_mean_change_ge_point25':int((np.abs(delta)>=.25).sum()),
                'genes_with_correct_sign_on_target_effect_ge_point25':int(((delta*target_delta>0)&meaningful).sum()),
                'genes_with_correct_sign_and_predicted_effect_ge_point25':int(((delta*target_delta>0)&meaningful&(np.abs(delta)>=.25)).sum()),
                'gene_mean_delta_correlation':corr,'zero_mask_preserved':bool(np.array_equal(pred==0,donors==0))})
        output['folds'].append(record)
    (HERE/'NEURAL_ODE_FORECAST_AUDIT.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
