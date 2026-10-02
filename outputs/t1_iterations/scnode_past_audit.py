"""Descriptive past reconstruction/transfer audit; no benchmark scoring."""
import json,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from scnode_joint_model import JointModel
from hurdle_backtest import read_cells
from run_t1 import digest
from iterate import append_event,now

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RUN=HERE/'private/scnode_past_audit_01';PUBLIC=HERE/'SCNODE_PAST_AUDIT_RESULTS.json'


def describe(model,values,seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        x=torch.tensor(values,dtype=torch.float32);mu,std=model.encode(x)
        decoded=model.decoder(mu)
        stochastic=model.decoder(mu+std*torch.randn_like(mu))
        final=model.trajectory(mu,torch.tensor([0.,1.]))[:,-1]
        future=model.decoder(final)
        zero=model.decoder(mu+0.*model.drift(mu))
        # Zero Euler step avoids duplicate solver times (invalid odeint).
        np.testing.assert_array_equal(zero.numpy(),decoded.numpy())
        drift=model.drift(mu)
    obs=values.astype(float);reconstruction=decoded.numpy().astype(float);delta=future.numpy().astype(float)-reconstruction
    var=obs.var(0);active=var>1e-8
    centered=obs-obs.mean(1,keepdims=True);other=reconstruction-reconstruction.mean(1,keepdims=True)
    correlations=(centered*other).sum(1)/np.maximum(np.sqrt((centered**2).sum(1)*(other**2).sum(1)),1e-12)
    result={'cells':len(values),'genes':values.shape[1],'reconstruction_mse':float(np.mean((reconstruction-obs)**2)),
            'stochastic_reconstruction_mse':float(((stochastic-x)**2).mean()),
            'mean_expression_mae':float(np.mean(abs(reconstruction.mean(0)-obs.mean(0)))),
            'observed_detection_fraction':float(np.mean(obs>0)),'decoded_detection_fraction':float(np.mean(reconstruction>0)),
            'observed_gene_variance_mean':float(var.mean()),'decoded_gene_variance_mean':float(reconstruction.var(0).mean()),
            'median_gene_variance_ratio':float(np.median(reconstruction.var(0)[active]/var[active])) if active.any() else None,
            'cell_expression_correlation_mean':float(correlations.mean()),
            'latent_mean_norm_mean':float(torch.linalg.vector_norm(mu,dim=1).mean()),
            'posterior_std_mean':float(std.mean()),'drift_norm_mean':float(torch.linalg.vector_norm(drift,dim=1).mean()),
            'elapsed_one_day_latent_change_norm_mean':float(torch.linalg.vector_norm(final-mu,dim=1).mean()),
            'one_day_decoded_abs_change_mean':float(np.mean(abs(delta))),
            'one_day_decoded_negative_fraction':float(np.mean(delta<0)),
            'one_day_decoded_positive_fraction':float(np.mean(delta>0)),
            'zero_elapsed_decoded_max_change':0.,
            'scope':'Past-fit diagnostic proxies; one-day unobserved decoded changes have no truth/comparison score.'}
    return result,mu.numpy().mean(0),mu.numpy().std(0)


def main():
    if RUN.exists() or PUBLIC.exists():raise ValueError('No duplicate audit')
    RUN.mkdir(parents=True);emit=lambda event,**kw:append_event(RUN/'events.jsonl',event,**kw)
    report={'status':'running','raw_metrics':None,'skills':None,'new_scoring_batch':False,'reward_delta':0,'passing_candidates':[]}
    try:
        source=HERE/'private/scnode_biological_pilot_01';prepared=HERE/'private/associated_prepared_01'
        spec_path=HERE/'NEXT_SCNODE_PAST_AUDIT.json';spec=json.loads(spec_path.read_text());prior=json.loads((source/'plan.json').read_text())
        for f,h in prior['source_sha256'].items():
            if digest(HERE/f)!=h:raise ValueError('At-run source mismatch: '+f)
        prep=json.loads((prepared/'report.json').read_text())
        for f,k in [('expression.npy','expression_sha256'),('genes.csv','genes_sha256'),('selected_metadata.csv','metadata_sha256')]:
            if digest(prepared/f)!=prep[k]:raise ValueError('Input mismatch')
        mapped=np.load(source/'mapped.npy');columns=np.load(source/'columns.npy');stages=pd.read_csv(prepared/'selected_metadata.csv').numeric_stage.to_numpy(float)
        x=np.load(prepared/'expression.npy',mmap_mode='r');rng=np.random.default_rng(spec['sampling_seed'])
        selected={}
        for stage in np.unique(stages[stages<=8.5]):
            eligible=np.flatnonzero(stages==stage);selected[str(stage)]=rng.choice(eligible,min(128,len(eligible)),replace=False)
            np.save(RUN/('source_rows_'+str(stage)+'.npy'),selected[str(stage)])
        panel=(ROOT/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines();donors,rows=read_cells(ROOT/'data/E8.5_RNA.h5ad',panel,1500,20260928)
        np.testing.assert_array_equal(rows,np.load(HERE/'private/cnf_covariance_alignment_01/donor_rows.npy'))
        anchor_ids=rng.choice(len(donors),128,replace=False);np.save(RUN/'anchor_sample_indices.npy',anchor_ids)
        plan={'spec_sha256':digest(spec_path),'prior_plan_sha256':digest(source/'plan.json'),
              'checkpoint_sha256':{name:digest(source/(name+'.pt')) for name in spec['arms']},
              'source_sha256':{f:digest(HERE/f) for f in ['scnode_past_audit.py','scnode_joint_model.py']},
              'mapped_sha256':digest(source/'mapped.npy'),'columns_sha256':digest(source/'columns.npy'),
              'seed':spec['sampling_seed'],'maximum_read_source_expression_stage':8.5,'anchor_stage':8.5,'no_future_expression':True}
        (RUN/'plan.json').write_text(json.dumps(plan,indent=2));report['plan_sha256']=digest(RUN/'plan.json');emit('plan_frozen',sha256=report['plan_sha256'])
        report['arms']={}
        for name in spec['arms']:
            saved=torch.load(source/(name+'.pt'),map_location='cpu',weights_only=False)
            if saved['fit_max_stage']!=8.5 or saved['steps']!=1000 or saved['seed']!=20261002 or saved['beta']!=float(name.split('_')[-1]):raise ValueError('Checkpoint fit metadata mismatch')
            model=JointModel(len(mapped));model.load_state_dict(saved['net']);model.eval();result={};latent={}
            for label,indices in selected.items():
                result[label],mean,std=describe(model,np.asarray(x[np.ix_(indices,columns)]),spec['sampling_seed'])
                latent[label]=(mean,std);emit('past_source_described',candidate=name,stage=float(label),**result[label])
            anchor,mean,std=describe(model,donors[np.ix_(anchor_ids,mapped)],spec['sampling_seed']);source_mean,source_std=latent['8.5']
            shift={'source_anchor_latent_centroid_distance':float(np.linalg.norm(mean-source_mean)),
                   'latent_source_std_norm':float(np.linalg.norm(source_std)),'latent_anchor_std_norm':float(np.linalg.norm(std)),
                   'centroid_shift_in_source_std_norms':float(np.linalg.norm(mean-source_mean)/max(np.linalg.norm(source_std),1e-12))}
            report['arms'][name]={'source_stages':result,'anchor':anchor,'domain_shift':shift,'training_final_record':saved['history'][-1]}
            emit('anchor_transfer_described',candidate=name,**anchor,**shift)
        report.update(status='completed',completed_utc=now(),scope='Training-fit/anchor diagnostics only. No reconstruction is held-out validation, no new four benchmark metrics, no causal official-score attribution; any earlier chronological fold needs fresh cutoff-trained representation.')
    except Exception as exc:
        (RUN/'traceback.txt').write_text(traceback.format_exc());report.update(status='failed',error=type(exc).__name__+': '+str(exc))
    (RUN/'report.json').write_text(json.dumps(report,indent=2));report['report_sha256']=digest(RUN/'report.json');PUBLIC.write_text(json.dumps(report,indent=2));emit('batch_finished',status=report['status']);print(json.dumps({'status':report['status']}))


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
