"""Post-scoring diagnosis only; never used to fit or choose frozen forecasts."""
import json
import numpy as np
from train_extended_atlas import HERE
from hurdle_backtest import read_cells
from offline_backtest import load_core
from run_t1 import digest
from iterate import now


def main():
    root=HERE.parents[1];out=HERE/'private/de_rank_trend_challenge_01'
    report=json.loads((out/'report.json').read_text())
    if report.get('status')!='completed':raise ValueError('Require completed frozen trial')
    panel=(root/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
    donor,rows=read_cells(root/'data/E8.5_RNA.h5ad',panel,1500,20260928)
    np.testing.assert_array_equal(rows,np.load(out/'anchor_rows.npy'))
    selected={}
    for k in [64,256]:
        with np.load(out/f'trend_k{k}.npz') as prior:
            selected[k]=(prior['genes'],prior['offset'])
    core,_=load_core();panels=[]
    for seed in report['plan']['seeds']:
        future,rows=read_cells(root/'data/E9.5_RNA.h5ad',panel,2000,seed)
        np.testing.assert_array_equal(rows,np.load(out/f'target_rows_{seed}.npy'))
        truth=future[np.random.default_rng(seed).permutation(len(future))[:1000]]
        up,down,_=core.de_genes(truth,donor)
        up=set(up.tolist());down=set(down.tolist())
        by_budget={}
        for k,(genes,offset) in selected.items():
            predicted_up=set(genes[offset>0].tolist());predicted_down=set(genes[offset<0].tolist())
            by_budget[str(k)]={'selected':len(genes),
                'true_DE_overlap':len(set(genes.tolist())&(up|down)),
                'directionally_correct_DE':len(predicted_up&up)+len(predicted_down&down),
                'directionally_wrong_DE':len(predicted_up&down)+len(predicted_down&up)}
        panels.append({'seed':seed,'n_up':len(up),'n_down':len(down),'selected_overlap':by_budget})
    result={'created_utc':now(),'scope':'Post-scoring E9.5 development diagnosis only; no candidate or model fitted from target labels.',
        'source_report_sha256':digest(out/'report.json'),'panels':panels,
        'not_independent_validation':True}
    (HERE/'DE_RANK_TREND_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
