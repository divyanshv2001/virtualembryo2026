"""Matched past-only potential transfer on reused released challenge development data."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event
from cnf_manifold_flow import DensityFlowNet, rk4_position
from graph_kinetic_residual import KineticFlow
from scalar_potential_residual import PotentialResidualFlow

MMD = '--past-mmd' in sys.argv
DIRECT = '--direct-cnf' in sys.argv or MMD
PREVIOUS = HERE / 'private/potential_real_anchor_transfer_02'
DIRECT_PREVIOUS = HERE / 'private/direct_cnf_potential_01'
RUN = HERE / ('private/past_multiscale_mmd_01' if MMD else 'private/direct_cnf_potential_01' if DIRECT else 'private/potential_real_anchor_transfer_02')
PUBLIC = HERE / ('PAST_MULTISCALE_MMD_RESULTS.json' if MMD else 'DIRECT_CNF_POTENTIAL_RESULTS.json' if DIRECT else 'POTENTIAL_REAL_ANCHOR_TRANSFER_REPAIR_RESULTS.json')


def past_bandwidth(coordinates):
    rng=np.random.default_rng(20261004)
    pairs=rng.integers(len(coordinates),size=(4096,2))
    distance=np.linalg.norm(coordinates[pairs[:,0]]-coordinates[pairs[:,1]],axis=1)
    value=float(np.median(distance[distance>0]))
    if not np.isfinite(value) or value<=0: raise ValueError('Invalid past bandwidth')
    return value


def multiscale_mmd(a,b,bandwidth):
    # Own unbiased latent RBF estimator; signed minibatch values are valid.
    if min(len(a),len(b))<2: raise ValueError('MMD needs two samples per set')
    def kernel(x,y):
        distance=(x[:,None,:]-y[None,:,:]).square().sum(-1)
        return torch.stack([torch.exp(-distance/(2*(bandwidth*s)**2)) for s in [.5,1.,2.,4.,8.]]).mean(0)
    aa,bb,ab=kernel(a,a),kernel(b,b),kernel(a,b)
    return (aa.sum()-aa.diagonal().sum())/(len(a)*(len(a)-1))+(bb.sum()-bb.diagonal().sum())/(len(b)*(len(b)-1))-2*ab.mean()


def drift(encoder, checkpoint, cutoff=8.5):
    model = DensityFlowNet(encoder['basis'], encoder['pca_center'], cutoff, 7.25)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['net'])
    return model.eval()


def train(packet_path):
    from geomloss import SamplesLoss
    packet = json.loads(Path(packet_path).read_text())
    for filename, sha in packet['sha256'].items():
        if digest(Path(filename)) != sha: raise ValueError('Frozen training input changed')
    if not torch.cuda.is_available() or torch.__version__ != '2.11.0+cu128' or '3060' not in torch.cuda.get_device_name(0):
        raise ValueError('Pinned RTX3060 runtime required')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.75)
    torch.use_deterministic_algorithms(True)
    context = dict(np.load(packet['context']))
    encoder = dict(np.load(packet['encoder']))
    times = np.unique(context['stages'])
    cutoff=packet.get('cutoff',8.5)
    expected_stages=packet.get('fit_stages',[7.5,7.75,8.,8.25,8.5])
    if times.tolist() != expected_stages or times.max()!=cutoff or context['coordinates'].shape != (packet.get('fit_rows',13963),8):
        raise ValueError('Past support changed')
    empty = np.array([], dtype=int)
    base = drift(encoder, packet['drift'],cutoff)
    if not DIRECT:
        base = KineticFlow(base, np.maximum(encoder['center']/encoder['scale'] + encoder['pca_center'], 0.), empty, empty, 'none')
    groups = [torch.tensor(context['coordinates'][context['stages'] == t], dtype=torch.float32, device='cuda') for t in times]
    objective = SamplesLoss('sinkhorn', p=2, blur=.05, scaling=.9, backend='tensorized')
    if MMD:
        if packet['bandwidth']!=past_bandwidth(context['coordinates']): raise ValueError('Frozen past bandwidth mismatch')
        objective=lambda a,b:multiscale_mmd(a,b,packet['bandwidth'])
    for kind in (['potential'] if DIRECT else ['kinetic', 'potential']):
        checkpoint = RUN / (kind + '400.pt')
        if checkpoint.exists(): raise ValueError('Never duplicate training')
        torch.manual_seed(20261004)
        model = (base if kind == 'kinetic' else PotentialResidualFlow(base, 'potential')).cuda()
        frozen_module = model.drift if kind == 'kinetic' else model.base
        frozen = {k: v.clone() for k, v in frozen_module.state_dict().items()}
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=.001)
        rng = torch.Generator().manual_seed(20261004)
        for step in range(400):
            j = int(torch.randint(len(groups)-1, (1,), generator=rng))
            a = groups[j][torch.randint(len(groups[j]), (64,), generator=rng).cuda()]
            b = groups[j+1][torch.randint(len(groups[j+1]), (64,), generator=rng).cuda()]
            optimizer.zero_grad()
            loss = objective(rk4_position(model.velocity, a, float(times[j]-7.25), float(times[j+1]-7.25), step=.125), b)
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward()
            if not torch.isfinite(torch.nn.utils.clip_grad_norm_(parameters, 5.)): raise ValueError('Nonfinite gradients')
            optimizer.step()
            if (step+1) % 50 == 0:
                append_event(RUN/'events.jsonl', 'training', arm=kind, step=step+1, loss=float(loss.detach().cpu()))
        for k, v in frozen.items(): torch.testing.assert_close(frozen_module.state_dict()[k], v, rtol=0, atol=0)
        peak = torch.cuda.max_memory_allocated()
        if peak > 4.5*1024**3: raise ValueError('GPU memory cap')
        model.cpu().eval()
        torch.save({'net': model.state_dict(), 'fit_max_stage': cutoff, 'steps': 400, 'kind': kind,
                    'seed': 20261004, 'frozen_base_exact': True, 'peak_allocated_bytes': peak,
                    'objective':'past_multiscale_mmd' if MMD else 'sinkhorn', 'bandwidth':packet.get('bandwidth')}, checkpoint)
        if kind == 'kinetic': base = model
        del optimizer, frozen
    for filename, sha in packet['sha256'].items():
        if digest(Path(filename)) != sha: raise ValueError('Training input changed during run')


def main():
    import anndata as ad
    from scipy import sparse
    from threadpoolctl import threadpool_limits
    from hurdle_backtest import read_cells
    from full_anchor_slope_forecast import FullAnchorSlopeForecast
    from temporary_forecast_cache import TemporaryForecastCache
    from frozen_fullpanel_scoring import score_frozen_forecasts
    from offline_backtest import load_core
    if RUN.exists() or PUBLIC.exists(): raise ValueError('Never overwrite prior evidence')
    root = HERE.parents[1]
    old = HERE/'private/cnf_feature_challenge_01'
    data = HERE/'private/associated_prepared_01'
    reference = json.loads((HERE/'private/cnf_full_anchor_challenge_01/report.json').read_text())
    old_plan = reference['plan']
    for filename, sha in old_plan['archive_sha256'].items():
        if 'cnf_feature_challenge' in filename and digest(HERE/filename) != sha: raise ValueError('Archive integrity')
    prepared = json.loads((data/'report.json').read_text())
    for name, key in [('expression.npy','expression_sha256'),('selected_metadata.csv','metadata_sha256'),('genes.csv','genes_sha256')]:
        if digest(data/name) != prepared[key]: raise ValueError('Prepared data integrity')
    panel = (root/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
    source_files = [Path(__file__), HERE/'scalar_potential_residual.py', HERE/'graph_kinetic_residual.py',
                    HERE/'cnf_manifold_flow.py', HERE/'full_anchor_slope_forecast.py', HERE/'offline_backtest.py',
                    HERE/'frozen_fullpanel_scoring.py', HERE/'hurdle_backtest.py']
    plan = {'created_utc': now(), 'cutoff': 8.5, 'target': 9.5, 'scoring_seeds': [20260928,20260929,20260930],
            'source_sha256': {str(p): digest(p) for p in source_files}, 'archive_sha256': old_plan['archive_sha256'],
            'input_sha256': old_plan['input_sha256'], 'training_seed':20261004, 'steps_per_arm':400,
            'hypothesis':('Does potential directly on frozen CNF improve transfer without the kinetic scaffold?' if DIRECT else 'Does the potential residual improve matched kinetics on actual challenge anchors?'),
            'scope':'Reused released E8.5->E9.5 development; single training seed, three scoring resamples. No independent embryos or hidden future fitting.',
            'heads':[.25,.75], 'fit_max_stage':8.5, 'source_rows':13963, 'panel_genes':32285,
            'target_row_order':'Adapter sorts each sampled set to preserve historical read_cells calibration exactly.',
            'official_score':None, 'promotion_gates_unchanged':True}
    plan['candidates']=['copy','cnf800','kinetic400','potential400']+(['direct_potential400'] if DIRECT else [])
    plan['primary_contrast']=['cnf800','direct_potential400'] if DIRECT else ['kinetic400','potential400']
    plan['training_base']='Frozen CNF800 directly' if DIRECT else 'Frozen CNF800 then kinetics400'
    if DIRECT:
        prior=json.loads((PREVIOUS/'report.json').read_text())
        plan['cached_controls_sha256']=prior['checkpoint_sha256']
        plan['cached_context_sha256']=digest(PREVIOUS/'context.npz')
        plan['previous_report_sha256']=digest(PREVIOUS/'report.json')
    if MMD:
        plan.update(hypothesis='Fixed past-only multiscale latentMMD versus Sinkhorn potential objective; same frozen CNF and training budget.',
                    candidates=['copy','cnf800','sinkhorn_direct400','mmd_direct400'],primary_contrast=['sinkhorn_direct400','mmd_direct400'],
                    bandwidth=past_bandwidth(np.load(PREVIOUS/'context.npz')['coordinates']),kernel_scales=[.5,1.,2.,4.,8.],
                    bandwidth_fit='Median of positive Euclidean distances of4096 seeded past-only random pairs; no future bandwidth fitting.',
                    mmd_estimator='Unbiased signed latent minibatch RBF mixture; unchanged official full-panel scorer.',
                    direct_control_report_sha256=digest(DIRECT_PREVIOUS/'report.json'))
    for filename, sha in plan['input_sha256'].items():
        if digest(root/filename) != sha: raise ValueError('Challenge input integrity')
    RUN.mkdir()
    (RUN/'plan.json').write_text(json.dumps(plan,indent=2))
    emit = lambda kind, **kw: append_event(RUN/'events.jsonl',kind,**kw)
    emit('plan_frozen', sha256=digest(RUN/'plan.json'))
    x = np.load(data/'expression.npy', mmap_mode='r')
    stages = pd.read_csv(data/'selected_metadata.csv').numeric_stage.to_numpy(float)
    symbols = pd.read_csv(data/'genes.csv').symbol.fillna('').tolist()
    encoder = dict(np.load(old/'encoder4096.npz'))
    past = np.flatnonzero(stages<=8.5)
    lookup = {s:i for i,s in enumerate(symbols)}
    columns = np.array([lookup[panel[i]] for i in encoder['features']])
    context_path=RUN/'context.npz'
    if DIRECT:
        previous_packet=json.loads((PREVIOUS/'training_packet.json').read_text())
        context_path=PREVIOUS/'context.npz'
        if digest(context_path)!=previous_packet['sha256'][str(context_path)]: raise ValueError('Cached context changed')
        previous_report=json.loads((PREVIOUS/'report.json').read_text())
        for name,sha in previous_report['checkpoint_sha256'].items():
            if digest(PREVIOUS/(name+'400.pt'))!=sha: raise ValueError('Cached control checkpoint changed')
    else:
        raw = np.asarray(x[np.ix_(past,columns)],np.float32)
        normalized = (raw-encoder['center'])/encoder['scale']
        coordinates = (normalized-encoder['pca_center'])@encoder['basis'].T
        np.savez_compressed(context_path,coordinates=coordinates,stages=stages[past])
        del raw,normalized,coordinates
    packet = {'context':str(context_path),'encoder':str(old/'encoder4096.npz'),'drift':str(old/'features4096.pt')}
    if MMD:
        packet['bandwidth']=plan['bandwidth']
        direct_reference=json.loads((DIRECT_PREVIOUS/'report.json').read_text())
        if digest(DIRECT_PREVIOUS/'potential400.pt')!=direct_reference['checkpoint_sha256']['potential']: raise ValueError('Direct control changed')
    packet['sha256'] = {str(p):digest(p) for p in source_files+[Path(packet[k]) for k in ['context','encoder','drift']]}
    (RUN/'training_packet.json').write_text(json.dumps(packet,indent=2))
    command=[str(root/'outputs/research_workflow/.venv_cuda/Scripts/python.exe'),str(Path(__file__)),
                    '--train-packet',str(RUN/'training_packet.json')]
    if DIRECT: command.append('--direct-cnf')
    if MMD: command.append('--past-mmd')
    subprocess.run(command,check=True)
    donors, rows = read_cells(root/'data/E8.5_RNA.h5ad',panel,1500,20260928)
    np.save(RUN/'anchor_rows.npy',rows)
    initial = drift(encoder,old/'features4096.pt')
    control_run=PREVIOUS if DIRECT else RUN
    saved = torch.load(control_run/'kinetic400.pt',map_location='cpu',weights_only=False)['net']
    empty = np.array([],dtype=int)
    base = KineticFlow(initial,saved['gene_mean'].numpy(),empty,empty,'none')
    base.load_state_dict(saved);base.eval()
    potential = PotentialResidualFlow(base,'potential')
    potential.load_state_dict(torch.load(control_run/'potential400.pt',map_location='cpu',weights_only=False)['net']);potential.eval()
    models=[('copy',None),('cnf800',initial),('kinetic400',base),('potential400',potential)]
    if DIRECT:
        direct=PotentialResidualFlow(initial,'potential')
        direct.load_state_dict(torch.load(RUN/'potential400.pt',map_location='cpu',weights_only=False)['net']);direct.eval()
        models.append(('direct_potential400',direct))
        if MMD:
            sinkhorn=PotentialResidualFlow(initial,'potential')
            sinkhorn.load_state_dict(torch.load(DIRECT_PREVIOUS/'potential400.pt',map_location='cpu',weights_only=False)['net']);sinkhorn.eval()
            models=[('copy',None),('cnf800',initial),('sinkhorn_direct400',sinkhorn),('mmd_direct400',direct)]
    cache = TemporaryForecastCache(HERE/'private/temporary_cache')
    generation = {}
    try:
        for name, model in models:
            if model is None: pred,indices,audit=donors.copy(),np.arange(len(donors)),{'method':'persistence'}
            else:
                program=FullAnchorSlopeForecast(x,stages,8.5,donors,panel,symbols,model,encoder['center'],encoder['scale'],
                    encoder['features'],encoder['guard_features'],anchor_path=root/'data/E8.5_RNA.h5ad')
                program.configure(.25,.75)
                if name==('mmd_direct400' if MMD else 'direct_potential400' if DIRECT else 'potential400'):
                    model.residual_enabled=False
                    baseline,_,_=program.predict(9.5,'joint',1.,sampling='systematic')
                    if cache.put('disabled_potential_replay',baseline)!=generation['cnf800' if DIRECT else 'kinetic400']['prediction_sha256']:
                        raise ValueError('Disabled potential exact kinetic replay failed')
                    with cache.read('disabled_potential_replay'): pass
                    del baseline
                    model.residual_enabled=True
                    emit('disabled_potential_exact_replay_passed')
                pred,indices,audit=program.predict(9.5,'joint',1.,sampling='systematic')
                del program
            generation[name]={'prediction_sha256':cache.put(name,pred),'audit':audit}
            np.save(RUN/(name+'_indices.npy'),indices)
            if name=='cnf800':
                expected=json.loads((HERE/'private/cnf_full_anchor_challenge_01/generation.json').read_text())['full_anchor_p0.25_d0.75']['prediction_sha256']
                if generation[name]['prediction_sha256']!=expected: raise ValueError('Archived CNF exact replay failed')
            if DIRECT and name in previous_report['generation']:
                if generation[name]['prediction_sha256']!=previous_report['generation'][name]['prediction_sha256']:
                    raise ValueError('Prior control forecast exact replay failed')
            if MMD and name=='sinkhorn_direct400' and generation[name]['prediction_sha256']!=direct_reference['generation']['direct_potential400']['prediction_sha256']:
                raise ValueError('Sinkhorn direct forecast exact replay failed')
            emit('forecast_frozen',candidate=name,sha256=generation[name]['prediction_sha256'])
            del pred
        (RUN/'generation.json').write_text(json.dumps(generation,indent=2))
        emit('all_forecasts_frozen_before_target_read')
        target=ad.read_h5ad(root/'data/E9.5_RNA.h5ad',backed='r')
        try:
            if target.var_names.tolist()!=panel: raise ValueError('Target gene order')
            class HistoricalSortedRows:
                rows=None
                values=None
                def __getitem__(self,index):
                    rr=np.sort(np.asarray(index[0]).ravel()); cc=np.asarray(index[1]).ravel()
                    if self.rows is None or not np.array_equal(self.rows,rr):
                        self.values=None
                        block=target.X[rr,:]
                        self.values=(block.toarray() if sparse.issparse(block) else np.asarray(block)).astype(np.float32)
                        if not np.isfinite(self.values).all() or (self.values<0).any(): raise ValueError('Invalid target values')
                        self.rows=rr
                    return self.values[:,cc]
            report={'plan':plan,'generation':generation,'panels':[], 'scorer_manifest_sha256':digest(HERE/'private/scorer_source/manifest.json')}
            core,_=load_core()
            score_frozen_forecasts(core,HistoricalSortedRows(),np.full(target.n_obs,9.5),donors,np.arange(len(panel)),np.arange(len(panel)),panel,
                8.5,9.5,plan,list(generation),generation,cache,RUN,report,emit,
                {'control_candidates':['copy','cnf800','sinkhorn_direct400'] if MMD else ['copy','cnf800','kinetic400','potential400'] if DIRECT else ['copy','cnf800','kinetic400'],
                 'contrast_candidates':['sinkhorn_direct400','mmd_direct400'] if MMD else ['cnf800','direct_potential400'] if DIRECT else ['kinetic400','potential400'],'scope':plan['scope']})
            for current,prior in zip(report['panels'],reference['panels']):
                if current['floor']!=prior['floor'] or current['ceiling']!=prior['ceiling']: raise ValueError('Historical calibration exact replay failed')
                cnf=next(r for r in current['results'] if r['candidate']=='cnf800')
                archived=next(r for r in prior['results'] if r['candidate']=='full_anchor_p0.25_d0.75')
                if cnf['raw_metrics']!=archived['raw_metrics']: raise ValueError('Historical CNF scorer exact replay failed')
                if DIRECT:
                    prior_panel=next(p for p in previous_report['panels'] if p['seed']==current['seed'])
                    for old_row in prior_panel['results']:
                        if MMD and old_row['candidate'] not in ['copy','cnf800']: continue
                        row=next(r for r in current['results'] if r['candidate']==old_row['candidate'])
                        if row['raw_metrics']!=old_row['raw_metrics']: raise ValueError('Prior control scorer exact replay failed')
                    if MMD:
                        prior_direct=next(p for p in direct_reference['panels'] if p['seed']==current['seed'])
                        expected=next(r for r in prior_direct['results'] if r['candidate']=='direct_potential400')['raw_metrics']
                        actual=next(r for r in current['results'] if r['candidate']=='sinkhorn_direct400')['raw_metrics']
                        if actual!=expected: raise ValueError('Sinkhorn direct scorer exact replay failed')
                np.save(RUN/f"target_rows_sorted_{current['seed']}.npy",np.sort(np.load(RUN/f"target_rows_{current['seed']}.npy")))
            report['checkpoint_sha256']={n:digest(RUN/(n+'400.pt')) for n in (['potential'] if DIRECT else ['kinetic','potential'])}
            report['local_72_gate_passed']=False
            if report['resource_peak_process_working_set_bytes']>16*1024**3: raise ValueError('Host memory cap')
            (RUN/'report.json').write_text(json.dumps(report,indent=2))
            report['report_sha256']=digest(RUN/'report.json')
            PUBLIC.write_text(json.dumps(report,indent=2))
            emit('completed',report_sha256=report['report_sha256'])
        finally: target.file.close()
    finally: cache.close()


if __name__=='__main__':
    if '--train-packet' in sys.argv: train(sys.argv[sys.argv.index('--train-packet')+1])
    else:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(2):
            torch.set_num_threads(2)
            try: main()
            except Exception as error:
                if RUN.exists() and not PUBLIC.exists():
                    failure={'status':'failed','error':str(error),'updated_utc':now(),'scope':'Matched actual-anchor transfer pilot; failure is not a validated score.'}
                    (RUN/'report.json').write_text(json.dumps(failure,indent=2))
                    failure['report_sha256']=digest(RUN/'report.json')
                    PUBLIC.write_text(json.dumps(failure,indent=2))
                raise
