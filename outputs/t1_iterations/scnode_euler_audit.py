"""Frozen past-only solver convergence diagnostic, not accuracy evaluation."""
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
RUN=HERE/'private/scnode_euler_audit_01';PUBLIC=HERE/'SCNODE_EULER_AUDIT_RESULTS.json'


def rms(x):return float(torch.sqrt(torch.mean(x*x)))


def main():
    if RUN.exists() or PUBLIC.exists():raise ValueError('No duplicate convergence audit')
    RUN.mkdir(parents=True);emit=lambda event,**kw:append_event(RUN/'events.jsonl',event,**kw)
    report={'status':'running','raw_metrics':None,'skills':None,'reward_delta':0,'new_scoring_batch':False,'passing_candidates':[]}
    try:
        spec_path=HERE/'NEXT_SCNODE_EULER_AUDIT.json';spec=json.loads(spec_path.read_text());source=HERE/spec['checkpoint_run'];past=HERE/spec['sample_indices_run'];prep=HERE/'private/associated_prepared_01'
        prior=json.loads((source/'plan.json').read_text());audit=json.loads((past/'plan.json').read_text())
        for f,h in prior['source_sha256'].items():
            if digest(HERE/f)!=h:raise ValueError('Biological at-run source changed')
        metadata=json.loads((prep/'report.json').read_text())
        for f,k in [('expression.npy','expression_sha256'),('genes.csv','genes_sha256'),('selected_metadata.csv','metadata_sha256')]:
            if digest(prep/f)!=metadata[k]:raise ValueError('Prepared input mismatch')
        mapped=np.load(source/'mapped.npy');columns=np.load(source/'columns.npy')
        if digest(source/'mapped.npy')!=audit['mapped_sha256'] or digest(source/'columns.npy')!=audit['columns_sha256']:raise ValueError('Gene mapping changed')
        indices=np.load(past/'source_rows_8.5.npy');stages=pd.read_csv(prep/'selected_metadata.csv').numeric_stage.to_numpy(float)
        if not np.all(stages[indices]==8.5):raise ValueError('Source sample stage mismatch')
        expression=np.load(prep/'expression.npy',mmap_mode='r');source_values=np.asarray(expression[np.ix_(indices,columns)],dtype=np.float32)
        panel=(ROOT/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines();donors,rows=read_cells(ROOT/'data/E8.5_RNA.h5ad',panel,1500,20260928)
        np.testing.assert_array_equal(rows,np.load(HERE/'private/cnf_covariance_alignment_01/donor_rows.npy'));anchor_indices=np.load(past/'anchor_sample_indices.npy')
        cohorts={'source8.5':source_values,'anchor8.5':donors[np.ix_(anchor_indices,mapped)]}
        plan={'spec_sha256':digest(spec_path),'prior_audit_plan_sha256':digest(past/'plan.json'),'source_sha256':{f:digest(HERE/f) for f in ['scnode_euler_audit.py','scnode_joint_model.py']},'checkpoint_sha256':{n:digest(source/(n+'.pt')) for n in spec['arms']},'sample_sha256':{f:digest(past/f) for f in ['source_rows_8.5.npy','anchor_sample_indices.npy']},'elapsed_days':1.,'substeps':spec['euler_substeps'],'no_future_expression':True}
        if plan['checkpoint_sha256']!=audit['checkpoint_sha256']:raise ValueError('Audit checkpoints changed')
        (RUN/'plan.json').write_text(json.dumps(plan,indent=2));report['plan_sha256']=digest(RUN/'plan.json');report['arms']={};emit('plan_frozen',sha256=report['plan_sha256'])
        for name in spec['arms']:
            saved=torch.load(source/(name+'.pt'),map_location='cpu',weights_only=False)
            if saved['fit_max_stage']!=8.5 or saved['steps']!=1000:raise ValueError('Checkpoint fit scope mismatch')
            model=JointModel(len(mapped));model.load_state_dict(saved['net']);model.eval();report['arms'][name]={}
            for cohort,values in cohorts.items():
                with torch.no_grad():
                    initial=model.encode(torch.tensor(values))[0];base=model.decoder(initial);latents={};decoded={};records=[]
                    for count in spec['euler_substeps']:
                        z=initial.clone()
                        for step in range(count):
                            z=z+model.drift(z)/count
                            if not torch.isfinite(z).all():raise ValueError('Nonfinite Euler state')
                        y=model.decoder(z)
                        if not torch.isfinite(y).all():raise ValueError('Nonfinite decoded state')
                        if count==1:torch.testing.assert_close(z,model.trajectory(initial,torch.tensor([0.,1.]))[:,-1],rtol=0.,atol=0.)
                        preactivation=model.decoder[2](model.decoder[:2](z))
                        latents[count]=z;decoded[count]=y
                        record={'substeps':count,'latent_displacement_rms':rms(z-initial),'decoded_change_rms':rms(y-base),
                                'decoded_gene_variance_mean':float(y.var(0,unbiased=False).mean()),
                                'decoded_zero_fraction':float((y==0).float().mean()),'decoder_preactivation_nonpositive_fraction':float((preactivation<=0).float().mean())}
                        records.append(record);emit('past_euler_step_described',candidate=name,cohort=cohort,**record)
                    differences=[]
                    for coarse,fine in [(1,4),(4,8),(8,16)]:
                        differences.append({'coarse':coarse,'fine':fine,'latent_relative_rms':rms(latents[coarse]-latents[fine])/max(rms(latents[fine]),1e-8),
                                            'decoded_change_relative_rms':rms(decoded[coarse]-decoded[fine])/max(rms(decoded[fine]-base),1e-8)})
                    passed=all(differences[-1][k]<=.05 for k in ['latent_relative_rms','decoded_change_relative_rms'])
                    report['arms'][name][cohort]={'steps':records,'differences':differences,'eight_vs_sixteen_converged':passed,
                                                'zero_time_decoded_variance_mean':float(base.var(0,unbiased=False).mean()),
                                                'observed_variance_mean':float(np.var(values,axis=0).mean()),'zero_step_max_change':0.}
        report.update(status='completed',completed_utc=now(),scope='Numerical convergence on exactly past-fit source/anchor samples; no accuracy/biological validation, no benchmark metrics or reward. Time-zero reconstruction unaffected by integration refinement.')
    except Exception as exc:
        (RUN/'traceback.txt').write_text(traceback.format_exc());report.update(status='failed',error=type(exc).__name__+': '+str(exc))
    (RUN/'report.json').write_text(json.dumps(report,indent=2));report['report_sha256']=digest(RUN/'report.json');PUBLIC.write_text(json.dumps(report,indent=2));emit('batch_finished',status=report['status']);print(json.dumps({'status':report['status']}))


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
