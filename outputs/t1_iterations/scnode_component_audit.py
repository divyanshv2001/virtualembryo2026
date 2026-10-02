"""Training-fit component diagnostics; deliberately no benchmark scoring."""
import sys,json,traceback,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
import torch
from threadpoolctl import threadpool_limits
from scnode_past_fold_training import JointModel,PastOnlyMatrix
from scnode_biological import AnchoredNeuralForecast
from scnode_past_fold import panel_values
from run_t1 import digest
from iterate import append_event,now

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RUN=HERE/'private/scnode_component_audit_01';PUBLIC=HERE/'SCNODE_COMPONENT_AUDIT_RESULTS.json'


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def alignment(delta,observed):
    active=abs(observed)>1e-6
    return {'mean_change_cosine':float(delta@observed/max(np.linalg.norm(delta)*np.linalg.norm(observed),1e-12)),
            'past_change_sign_agreement_fraction':float(np.mean(np.sign(delta[active])==np.sign(observed[active]))) if active.any() else None,
            'active_observed_genes':int(active.sum())}


def main():
    if RUN.exists() or PUBLIC.exists():raise ValueError('No duplicate audit')
    RUN.mkdir(parents=True);emit=lambda event,**kw:append_event(RUN/'events.jsonl',event,**kw)
    report={'status':'running','raw_metrics':None,'skills':None,'reward_delta':0,'new_scoring_batch':False,'passing_candidates':[]}
    try:
        spec_path=HERE/'NEXT_SCNODE_COMPONENT_AUDIT.json';spec=json.loads(spec_path.read_text())
        source=HERE/spec['source_run'];prior=json.loads((source/'plan.json').read_text())
        for f,h in prior['source_sha256'].items():
            if digest(HERE/f)!=h:raise ValueError('Frozen source changed: '+f)
        prepared=HERE/'private/associated_prepared_01'
        for f,h in prior['input_sha256'].items():
            if digest(prepared/f)!=h:raise ValueError('Prepared input changed: '+f)
        plan={'created_utc':now(),'spec_sha256':digest(spec_path),'prior_plan_sha256':digest(source/'plan.json'),
              'checkpoint_sha256':{n:digest(source/(n+'.pt')) for n in spec['arms']},
              'source_sha256':{f:digest(HERE/f) for f in ['scnode_component_audit.py','scnode_past_fold_training.py','scnode_biological.py']},
              'max_expression_stage':8.0,'scope':'In-sample descriptive, training already saw endpoint8.0; no scorer or predictive validation'}
        save(RUN/'plan.json',plan);emit('plan_frozen',sha256=digest(RUN/'plan.json'))
        stages=pd.read_csv(prepared/'selected_metadata.csv').numeric_stage.to_numpy(float)
        x=PastOnlyMatrix(np.load(prepared/'expression.npy',mmap_mode='r'),stages,8.)
        mapped=np.load(source/'mapped.npy');columns=np.load(source/'columns.npy')
        with np.load(source/'fresh_encoder.npz') as e:guard=e['guard_features'].copy()
        panel=(ROOT/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
        rng=np.random.default_rng(spec['seed']);selected={};means={};observed={}
        for stage in spec['source_stages']:
            rows=np.flatnonzero(stages==stage);ids=rng.choice(rows,128,replace=False)
            selected[stage]=ids;np.save(RUN/f'rows_{stage}.npy',ids)
            observed[stage]=np.asarray(x[np.ix_(ids,columns)],np.float32)
            mean=np.empty(len(columns))
            for start in range(0,len(columns),512):mean[start:start+512]=np.asarray(x[np.ix_(rows,columns[start:start+512])],float).mean(0)
            means[stage]=mean
        donor_rows=selected[7.75];donors=panel_values(x,donor_rows,mapped,columns,len(panel))
        anchor_rows=np.flatnonzero(stages==7.75);anchor_path=RUN/'source_E7.75_anchor.h5ad'
        anchor=ad.AnnData(sparse.csr_matrix(panel_values(x,anchor_rows,mapped,columns,len(panel))),
                         obs=pd.DataFrame(index=[str(i) for i in anchor_rows]),var=pd.DataFrame(index=panel))
        anchor.write_h5ad(anchor_path);del anchor
        past_delta=means[8.]-means[7.75];protected=np.setdiff1d(np.arange(len(panel)),mapped)
        report['arms']={}
        for name in spec['arms']:
            saved=torch.load(source/(name+'.pt'),weights_only=False,map_location='cpu')
            if saved['fit_max_stage']!=8. or saved['steps']!=1000 or saved['seed']!=20261002:raise ValueError('Checkpoint boundary mismatch')
            model=JointModel(len(columns));model.load_state_dict(saved['net']);model.eval()
            descriptions={}
            with torch.no_grad():
                for stage,values in observed.items():
                    tensor=torch.tensor(values);mu,_=model.encode(tensor);decoded=model.decoder(mu).numpy();drift=model.drift(mu).numpy()
                    var=values.astype(float).var(0);dv=decoded.astype(float).var(0)
                    descriptions[str(stage)]={'reconstruction_mse':float(((decoded-values)**2).mean()),
                        'observed_detection_fraction':float((values>0).mean()),'decoded_detection_fraction':float((decoded>0).mean()),
                        'observed_gene_variance_mean':float(var.mean()),'decoded_gene_variance_mean':float(dv.mean()),
                        'decoded_to_observed_mean_variance_ratio':float(dv.mean()/max(var.mean(),1e-12)),
                        'negative_latent_velocity_fraction':float((drift<0).mean()),'zero_latent_velocity_fraction':float((drift==0).mean())}
                mu,_=model.encode(torch.tensor(observed[7.75]));anchor_decoded=model.decoder(mu)
                final=model.trajectory(mu,torch.tensor([0.,.25]))[:,-1]
                decoded_delta=(model.decoder(final)-anchor_decoded).mean(0).numpy().astype(float)
            program=AnchoredNeuralForecast(x,stages,8.,donors,mapped,columns,model,guard,anchor_path)
            # This explicitly replays an IN-SAMPLE earlier interval; all weights/heads saw8.0.
            program.cutoff=7.75
            components={}
            for mode in spec['modes']:
                pred,ids,audit=program.predict(8.,mode,1.,sampling='systematic',seed=20260928)
                np.testing.assert_array_equal(pred[:,protected],donors[:,protected])
                if not np.isfinite(pred).all() or (pred<0).any():raise ValueError('Invalid component')
                mass0=np.expm1(donors[:,mapped].astype(float)).sum(1);mass1=np.expm1(pred[:,mapped].astype(float)).sum(1)
                np.testing.assert_allclose(mass0,mass1,rtol=1e-5,atol=1e-4)
                delta=pred[:,mapped].mean(0,dtype=float)-donors[:,mapped].mean(0,dtype=float)
                components[mode]={'audit':audit,'prediction_array_sha256':hashlib.sha256(pred.tobytes()).hexdigest(),
                                  'alignment_to_observed_training_transition':alignment(delta,past_delta),
                                  'mapped_mass_max_relative_error':float(np.max(abs(mass1-mass0)/np.maximum(mass0,1e-12))),
                                  'observed_donor_detection_fraction':float((donors[:,mapped]>0).mean()),
                                  'component_detection_fraction':float((pred[:,mapped]>0).mean())}
                emit('past_component_described',candidate=name,mode=mode,**components[mode]);del pred
            report['arms'][name]={'reconstruction':descriptions,'decoded_transition_alignment':alignment(decoded_delta,past_delta),'components':components}
            del program,model
        report.update(status='completed',completed_utc=now(),scope=plan['scope'],plan_sha256=digest(RUN/'plan.json'))
    except Exception as exc:
        (RUN/'traceback.txt').write_text(traceback.format_exc());report.update(status='failed',error=type(exc).__name__+': '+str(exc))
    save(RUN/'report.json',report);report['report_sha256']=digest(RUN/'report.json');save(PUBLIC,report)
    emit('batch_finished',status=report['status']);print(json.dumps({'status':report['status']}))

if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
