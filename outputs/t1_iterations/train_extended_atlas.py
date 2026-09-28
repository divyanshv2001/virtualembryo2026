"""Train time-conditioned latent velocity models on permitted early atlas stages.

Forecasts are scored on held-out atlas stages; no official score or upload capability.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from iterate import append_event,now
from offline_backtest import load_core,Panel
sys.path.insert(0,str(HERE.parents[1]/'outputs/t1_run'))
from run_t1 import digest
DATA = HERE/'private/extended_prepared_02'
OUT = HERE/'private/extended_training_01'


def normalize_log(x):
    abundance = np.expm1(np.maximum(x,0).astype(np.float64))
    sums = abundance.sum(1)
    if not np.isfinite(abundance).all() or (sums<=0).any(): raise ValueError('Invalid implied libraries')
    return np.log1p(abundance*(10000/sums[:,None])).astype(np.float32)


class Dynamics:
    def __init__(self, x, stages, max_stage, events):
        self.stages = stages
        self.rows = np.flatnonzero(stages<=max_stage)
        if not len(self.rows): raise ValueError('No training cells')
        self.x = x; self.max_stage = max_stage
        n_genes = x.shape[1]; variance = np.zeros(n_genes)
        for start in range(0,n_genes,512):
            block = np.asarray(x[self.rows,start:start+512],dtype=np.float64)
            variance[start:start+512] = block.var(0)
        self.features = np.sort(np.argsort(-variance,kind='stable')[:768])
        values = np.asarray(x[np.ix_(self.rows,self.features)],dtype=np.float64)
        self.center = values.mean(0); self.scale = np.maximum(values.std(0),.1)
        self.pca = PCA(n_components=24,whiten=True,random_state=20260928)
        z = self.pca.fit_transform(np.clip((values-self.center)/self.scale,-10,10))
        self.z = z; local_stages = stages[self.rows]
        self.z_mean = z.mean(0)
        # Whole-panel decoder is fitted only on training rows. Preserve donor residuals in forecasts.
        gram = z.T@z + 100*np.eye(z.shape[1])
        self.decoder = np.empty((z.shape[1],n_genes),dtype=np.float64)
        self.gene_mean = np.empty(n_genes,dtype=np.float64)
        for start in range(0,n_genes,512):
            block = np.asarray(x[self.rows,start:start+512],dtype=np.float64)
            center = block.mean(0); self.gene_mean[start:start+512] = center
            self.decoder[:,start:start+512] = np.linalg.solve(gram,z.T@(block-center))
        unique = np.unique(local_stages); states = []; times = []; velocities = []
        for a,b in zip(unique[:-1],unique[1:]):
            early = z[local_stages==a]; late = z[local_stages==b]
            _,indices = cKDTree(early).query(late,k=16,workers=2)
            matched = early[indices].mean(1)
            states.append(matched); times.append(np.full(len(matched),a))
            velocities.append((late-matched)/(b-a))
        self.states = np.vstack(states); self.times = np.concatenate(times); self.velocities = np.vstack(velocities)
        self.speed_cap = float(np.quantile(np.linalg.norm(self.velocities,axis=1),.95))
        rng = np.random.default_rng(20260928)
        self.rff_weights = rng.normal(size=(24,128))/np.sqrt(24)
        self.rff_bias = rng.uniform(0,2*np.pi,128)
        self.models = {}
        append_event(events,'representation_and_decoder_fitted',max_stage=max_stage,training_cells=len(self.rows),
            features=len(self.features),dimensions=24,pseudo_pairs=len(self.states),speed_cap=self.speed_cap)

    def phi(self,z,t,nonlinear):
        time = np.asarray(t).reshape(-1,1)-7.
        basic = np.concatenate([z,time,z*time],axis=1)
        if nonlinear: return np.concatenate([basic,np.cos(z@self.rff_weights+self.rff_bias)],axis=1)
        return basic

    def fit_velocity(self,regularization,nonlinear,events):
        key = (regularization,nonlinear)
        if key not in self.models:
            features = self.phi(self.states,self.times,nonlinear)
            model = Ridge(alpha=regularization).fit(features,self.velocities)
            self.models[key] = model
            residual = model.predict(features)-self.velocities
            append_event(events,'velocity_model_trained',max_stage=self.max_stage,regularization=regularization,
                nonlinear=nonlinear,training_pseudo_pair_rmse=float(np.sqrt(np.mean(residual**2))),
                scope='In-sample pseudo-pair fit only; not forecast validation')
        return self.models[key]

    def forecast(self,config,target,events):
        last_rows = np.flatnonzero(self.stages==self.max_stage)
        donor = np.asarray(self.x[last_rows],dtype=np.float32)
        if config['method']=='copy_last': return donor
        if config['method']=='mean_trend':
            used_stages = np.unique(self.stages[self.rows])[-3:]
            means = np.stack([np.asarray(self.x[np.flatnonzero(self.stages==t)]).mean(0,dtype=np.float64) for t in used_stages])
            centered = used_stages-used_stages.mean()
            slope = centered@means/np.sum(centered**2)
            return normalize_log(donor+config['strength']*(target-self.max_stage)*slope)
        model = self.fit_velocity(config['ridge'],config['nonlinear'],events)
        values = np.asarray(self.x[np.ix_(last_rows,self.features)],dtype=np.float64)
        z = self.pca.transform(np.clip((values-self.center)/self.scale,-10,10)); original_z = z.copy()
        t = self.max_stage
        while t<target-1e-8:
            dt = min(.25,target-t)
            v = model.predict(self.phi(z,np.full(len(z),t),config['nonlinear']))
            speed = np.linalg.norm(v,axis=1)
            v *= np.minimum(1,self.speed_cap/np.maximum(speed,1e-9))[:,None]
            z += dt*config['strength']*v; t += dt
        prediction = donor.astype(np.float64)+(z-original_z)@self.decoder
        return normalize_log(prediction)

    def save(self,path):
        arrays = {'features':self.features,'center':self.center,'scale':self.scale,
            'pca_components':self.pca.components_,'pca_mean':self.pca.mean_,
            'pca_explained_variance':self.pca.explained_variance_,'decoder':self.decoder,
            'gene_mean':self.gene_mean,'rff_weights':self.rff_weights,'rff_bias':self.rff_bias,
            'speed_cap':np.array(self.speed_cap),'max_training_stage':np.array(self.max_stage)}
        for i,((reg,nonlinear),model) in enumerate(self.models.items()):
            arrays[f'model_{i}_coef'] = model.coef_; arrays[f'model_{i}_intercept'] = model.intercept_
            arrays[f'model_{i}_config'] = np.array([reg,int(nonlinear)])
        np.savez_compressed(path,**arrays)


def evaluate_stage(core,x,stages,reference,predictions,target,events):
    rows = np.flatnonzero(stages==target)
    # Random-cell halves define a sampling ceiling, not independent embryo replication.
    shuffled = np.random.default_rng(20260928).permutation(rows)
    truth, ceiling_rows = np.array_split(shuffled,2)
    panel = Panel(core,np.asarray(x[truth]),reference,20260928)
    floor = panel.metrics(reference); ceiling = panel.metrics(np.asarray(x[ceiling_rows]))
    result = []
    for name,prediction in predictions.items():
        raw = panel.metrics(prediction)
        row = {'candidate':name,'raw_metrics':raw,**panel.aggregate(raw,floor,ceiling)}
        result.append(row)
        append_event(events,'atlas_candidate_evaluated',target=target,candidate=name,
            local_score=row['local_score'],calibration_valid=row['calibration_valid'])
    return {'target_stage':target,'scope':'Full atlas gene panel; local stage-specific anchors, not official challenge score.',
        'floor':floor,'ceiling':ceiling,'truth_cells':len(truth),'ceiling_cells':len(ceiling_rows),
        'ceiling_replication':'Random cell halves; sample IDs are not asserted to be independent embryo IDs.',
        'results':result}


def main():
    if OUT.exists(): raise SystemExit('Training run exists; preserve its selection/test history.')
    prepared = json.loads((DATA/'report.json').read_text())
    if digest(DATA/'expression.npy')!=prepared['expression_sha256']: raise ValueError('Prepared expression changed')
    if digest(DATA/'selected_metadata.csv')!=prepared['metadata_sha256']: raise ValueError('Metadata changed')
    OUT.mkdir(parents=True); events = OUT/'events.jsonl'
    configs = [{'name':'copy_last','method':'copy_last'}]
    configs += [{'name':f'mean_trend_{s}','method':'mean_trend','strength':s} for s in [.5,1.]]
    configs += [{'name':f'velocity_{"rff" if nonlinear else "linear"}_r{reg}_s{s}',
        'method':'latent_velocity','ridge':reg,'nonlinear':nonlinear,'strength':s}
        for nonlinear,reg in [(False,10.),(False,100.),(True,100.)] for s in [.5,1.]]
    plan = {'created_before_training_utc':now(),'development_fit_max_stage':7.5,'development_forecast_stage':8.5,
        'final_fit_max_stage':8.5,'final_forecast_stage':9.5,'configs':configs,
        'feature_selection':'768 highest training-only variance genes; target stages never used for fitting.',
        'embeddings':'Training-only PCA; supplied atlas PCA/UMAP/clusters not used.',
        'coupling':'Nearest-neighbor barycentric pseudo-pairs, not known lineage or an OT solver.',
        'selection':'Highest calibrated development score; final stage evaluated once after selecting and refitting.',
        'score_scope':'Earlier-stage atlas proxy only; no hidden E10.5 score prediction.',
        'normalization':'Natural log1p, total implied library 10000 across complete atlas gene panel.',
        'code_sha256':digest(Path(__file__)),'data_report_sha256':digest(DATA/'report.json'),
        'submissions_allowed':0,'jev_requests_allowed':0}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    (OUT/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    append_event(events,'training_plan_frozen',sha256=digest(OUT/'plan.json'))
    metadata = pd.read_csv(DATA/'selected_metadata.csv',low_memory=False)
    stages = metadata.numeric_stage.to_numpy(dtype=float)
    x = np.load(DATA/'expression.npy',mmap_mode='r')
    development = Dynamics(x,stages,7.5,events)
    predictions = {c['name']:development.forecast(c,8.5,events) for c in configs}
    development.save(OUT/'development_checkpoint.npz')
    append_event(events,'development_forecasts_frozen',candidates=list(predictions),training_max_stage=7.5)
    core,_ = load_core()
    development_metrics = evaluate_stage(core,x,stages,np.asarray(x[stages==7.5]),predictions,8.5,events)
    valid = [r for r in development_metrics['results'] if r['calibration_valid']]
    report = {'plan':plan,'development':development_metrics,'official_score':None,'official_72_verified':False,
        'submissions_used':0,'jev_requests_used':0,'prepared_data':prepared,
        'limitations':['Broad whole-embryo atlas differs from challenge dissection.',
            'Local calibration/atlas gene panel differ from official T1 panel.',
            'Pseudo-pair training errors do not measure biological lineage accuracy.',
            'Only one development and one final stage; no independent embryo confidence interval.']}
    if not valid:
        report.update(status='development_calibration_failed',selected=None,final=None)
    else:
        winner = max(valid,key=lambda r:r['local_score'])
        config = next(c for c in configs if c['name']==winner['candidate'])
        (OUT/'selection.json').write_text(json.dumps({'config':config,'development_result':winner,
            'final_not_yet_evaluated':True},indent=2),encoding='utf-8')
        append_event(events,'selected_before_final_stage_evaluation',candidate=config['name'],development_local_score=winner['local_score'])
        del development,predictions
        final = Dynamics(x,stages,8.5,events)
        prediction = final.forecast(config,9.5,events)
        final.save(OUT/'final_checkpoint.npz')
        final_metrics = evaluate_stage(core,x,stages,np.asarray(x[stages==8.5]),{config['name']:prediction},9.5,events)
        result = final_metrics['results'][0]
        local_gate = bool(winner['local_score']>72 and result['calibration_valid'] and result['local_score']>72)
        report.update(status='completed',selected=config,final=final_metrics,local_proxy_72_gate_passed=local_gate)
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    append_event(events,'training_run_completed',status=report['status'],official_72_verified=False)


if __name__=='__main__':
    with threadpool_limits(limits=2): main()
