"""Observed E8.5 full-gene decoder ablation on a frozen historical CNF path."""
import json
import sys
import subprocess
from pathlib import Path
import numpy as np
import torch
from train_extended_atlas import HERE
from run_t1 import digest
from iterate import now, append_event

RUN = HERE / 'private/real_fullgene_decoder_pilot_03'
PUBLIC = HERE / 'REAL_FULLGENE_DECODER_PILOT_RECOVERY_RESULTS.json'


def head():
    return torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.Tanh(),
                               torch.nn.Linear(64, 32285))


def train():
    if not torch.cuda.is_available():
        raise RuntimeError('Declared RTX3060 unavailable; no CPU fallback')
    torch.cuda.set_per_process_memory_fraction(.75)
    torch.set_num_threads(2)
    torch.manual_seed(20261004)
    torch.cuda.manual_seed_all(20261004)
    torch.use_deterministic_algorithms(True)
    packet = json.loads((RUN / 'training.json').read_text())
    for filename, expected in packet['sha256'].items():
        if digest(Path(filename)) != expected:
            raise ValueError('Training input changed')
    y = np.load(RUN / 'observed_expression.npy', mmap_mode='r')
    z = np.load(RUN / 'coordinates.npy')
    center, scale = z.mean(0), np.maximum(z.std(0), .1)
    z = (z-center)/scale
    mean = np.load(RUN / 'gene_mean.npy')
    net = head().cuda()
    with torch.no_grad():
        net[-1].weight.zero_()
        net[-1].bias.copy_(torch.tensor(mean, device='cuda'))
    opt = torch.optim.Adam(net.parameters(), lr=.001)
    rng = np.random.default_rng(20261004)
    for step in range(400):
        rows = rng.choice(len(z), 128, replace=False)
        x = torch.tensor(z[rows], device='cuda')
        target = torch.tensor(np.asarray(y[rows]), device='cuda')
        opt.zero_grad(set_to_none=True)
        loss = (net(x)-target).square().mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite decoder loss')
        loss.backward()
        if not all(torch.isfinite(v.grad).all() for v in net.parameters()):
            raise ValueError('Nonfinite decoder gradients')
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
        opt.step()
        if step in (0, 99, 199, 299, 399):
            append_event(RUN/'events.jsonl', 'cuda_decoder_step', step=step+1,
                         loss=float(loss.detach()), allocated=torch.cuda.max_memory_allocated())
    torch.save({'net':net.cpu().state_dict(), 'center':center, 'scale':scale,
                'fit_stage':8.5, 'fit_rows':len(z), 'steps':400,
                'device':torch.cuda.get_device_name(0),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated()}, RUN/'decoder.pt')


def main():
    import anndata as ad
    import pandas as pd
    from scnode_resource_preflight import peak_memory
    from scipy import sparse
    from threadpoolctl import threadpool_limits
    from cnf_manifold_flow import DensityFlowNet
    from full_anchor_slope_forecast import FullAnchorSlopeForecast
    from cnf_covariance_alignment import CovarianceAlignedTrajectory
    from temporary_forecast_cache import TemporaryForecastCache
    from hurdle_backtest import read_cells
    from offline_backtest import load_core, Panel
    if RUN.exists() or PUBLIC.exists():
        raise ValueError('Never duplicate this trial')
    root = HERE.parents[1]
    archive = HERE/'private/cnf_feature_challenge_01'
    alignment = HERE/'private/cnf_covariance_alignment_01'
    data = HERE/'private/associated_prepared_01'
    anchor = root/'data/E8.5_RNA.h5ad'
    panel = (root/'outputs/t1_run/T1__val.genes.txt').read_text().splitlines()
    plan = {'created_utc':now(), 'cutoff':8.5, 'target':9.5,
            'training_rows':16787, 'genes':32285, 'steps':400, 'seed':20261004,
            'scoring_seeds':[20260928,20260929,20260930],
            'candidates':['copy','incumbent','linear_fullgene','neural_fullgene'],
            'primary_contrast':['linear_fullgene','neural_fullgene'],
            'fit':'Full real E8.5 only; frozen CNF latent trajectory and .25 covariance map. New 8->64 tanh->32285 MSE decoder, batch128 Adam.001. Linear ridge1 fullgene control. Add clipped[-.25,.25] conditional decoder difference to the same original donor, restore full library mass. No E9.5 fit or labels.',
            'hypothesis':'Real-anchor nonlinear full-transcriptome conditional decoding improves beyond linear decoding and archived source decoder without changing dynamics.',
            'scope':'Previously exposed real E9.5 development; one training seed and three scoring resamples, not independent embryo validation.',
            'readiness_gates_unchanged':True,'official_uploads':0,
            'sources':{str(f):digest(f) for f in [Path(__file__), archive/'encoder4096.npz', archive/'features4096.pt', alignment/'alignment.npz', alignment/'report.json', HERE/'offline_backtest.py', root/'outputs/t1_run/T1__val.genes.txt']}}
    RUN.mkdir()
    (RUN/'plan.json').write_text(json.dumps(plan,indent=2))
    (RUN/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    emit = lambda event, **kw:append_event(RUN/'events.jsonl',event,**kw)
    emit('plan_frozen',sha256=digest(RUN/'plan.json'))
    with threadpool_limits(limits=2):
        enc = dict(np.load(archive/'encoder4096.npz'))
        net = DensityFlowNet(enc['basis'],enc['pca_center'],8.5,7.25)
        net.load_state_dict(torch.load(archive/'features4096.pt',weights_only=False,map_location='cpu')['net'])
        net.eval()
        donors, ids = read_cells(anchor,panel,1500,20260928)
        np.testing.assert_array_equal(ids,np.load(alignment/'donor_rows.npy'))
        a = ad.read_h5ad(anchor,backed='r')
        if a.var_names.tolist()!=panel or a.n_obs!=16787:
            raise ValueError('Observed panel or row count changed')
        y = np.lib.format.open_memmap(RUN/'observed_expression.npy',mode='w+',dtype='float32',shape=a.shape)
        z = np.empty((a.n_obs,8),dtype=np.float32)
        try:
            for start in range(0,a.n_obs,256):
                block = a.X[start:start+256]
                block = block.toarray() if sparse.issparse(block) else np.asarray(block)
                y[start:start+len(block)] = block
                with torch.no_grad():
                    z[start:start+len(block)] = net.encode(torch.tensor((block[:,enc['features']]-enc['center'])/enc['scale']))[0].numpy()
                if peak_memory()>16*1024**3:
                    raise MemoryError('Host memory ceiling exceeded')
        finally:
            a.file.close()
        y.flush()
        mean = np.zeros(32285,dtype=np.float64)
        zc = z.astype(np.float64)-z.mean(0)
        cross = np.zeros((8,32285),dtype=np.float64)
        for start in range(0,len(z),256):
            block=np.asarray(y[start:start+256],dtype=np.float64)
            mean+=block.sum(0)
            cross+=zc[start:start+len(block)].T@block
        mean/=len(z)
        coef=np.linalg.solve(zc.T@zc/len(z)+np.eye(8),cross/len(z))
        np.save(RUN/'gene_mean.npy',mean.astype(np.float32))
        np.save(RUN/'coordinates.npy',z)
        del y
        files=[RUN/'observed_expression.npy',RUN/'coordinates.npy',RUN/'gene_mean.npy',Path(__file__)]
        (RUN/'training.json').write_text(json.dumps({'sha256':{str(f):digest(f) for f in files}}))
        cuda=root/'outputs/research_workflow/.venv_cuda/Scripts/python.exe'
        prior=HERE/'private/real_fullgene_decoder_pilot_02'
        if (prior/'decoder.pt').exists():
            old_packet=json.loads((prior/'training.json').read_text())
            for filename in ['observed_expression.npy','coordinates.npy','gene_mean.npy']:
                if digest(RUN/filename)!=old_packet['sha256'][str(prior/filename)]:
                    raise ValueError('Recovery training data changed')
            saved=torch.load(prior/'decoder.pt',weights_only=False,map_location='cpu')
            if saved['fit_stage']!=8.5 or saved['fit_rows']!=16787 or saved['steps']!=400:
                raise ValueError('Recovery checkpoint scope mismatch')
            emit('cuda_checkpoint_reused',sha256=digest(prior/'decoder.pt'),original_run=str(prior))
        else:
            subprocess.run([str(cuda),'-u',str(Path(__file__)),'--train'],check=True)
            saved=torch.load(RUN/'decoder.pt',weights_only=False,map_location='cpu')
        decoder=head();decoder.load_state_dict(saved['net']);decoder.eval()
        al=dict(np.load(alignment/'alignment.npz'))
        transform=np.eye(8)+.25*(al['full_map']-np.eye(8))
        field=CovarianceAlignedTrajectory(net,al['source_mean'],al['challenge_mean'],transform)
        with torch.no_grad():
            z0=net.encode(torch.tensor((donors[:,enc['features']]-enc['center'])/enc['scale']))[0]
            z1=field.trajectory(z0,torch.tensor([0.,1.]))[-1]
            change=decoder((z1-torch.tensor(saved['center']))/torch.tensor(saved['scale']))-decoder((z0-torch.tensor(saved['center']))/torch.tensor(saved['scale']))
        cache=TemporaryForecastCache(HERE/'private/temporary_cache')
        try:
            generation={'copy':{'prediction_sha256':cache.put('copy',donors)}}
            for name,delta in [('linear_fullgene',(z1.numpy()-z0.numpy())@coef),('neural_fullgene',change.numpy())]:
                pred=np.maximum(donors+np.clip(delta,-.25,.25),0).astype(np.float32)
                mass=np.expm1(donors.astype(np.float64)).sum(1)
                abundance=np.expm1(pred.astype(np.float64))
                pred=np.log1p(abundance*(mass/np.maximum(abundance.sum(1),1e-12))[:,None]).astype(np.float32)
                if not np.isfinite(pred).all():raise ValueError('Nonfinite forecast')
                generation[name]={'prediction_sha256':cache.put(name,pred)}
                emit('forecast_frozen',candidate=name,**generation[name])
                del pred,abundance
            x=np.load(data/'expression.npy',mmap_mode='r')
            stages=pd.read_csv(data/'selected_metadata.csv').numeric_stage.to_numpy(float)
            symbols=pd.read_csv(data/'genes.csv').symbol.fillna('').tolist()
            program=FullAnchorSlopeForecast(x,stages,8.5,donors,panel,symbols,net,enc['center'],enc['scale'],enc['features'],np.load(archive/'features.npy'),anchor_path=anchor)
            program.configure(.25,.75)
            program.net=field
            pred,_,_=program.predict(9.5,'joint',1.,sampling='systematic')
            generation['incumbent']={'prediction_sha256':cache.put('incumbent',pred)}
            del pred
            archived=json.loads((alignment/'generation.json').read_text())
            if generation['incumbent']['prediction_sha256']!=archived['covariance_0.25']['prediction_sha256']:
                raise ValueError('Incumbent exact replay failed')
            emit('all_forecasts_frozen',generation=generation)
            core,_=load_core();report={'status':'scoring','plan':plan,'panels':[]}
            old=json.loads((alignment/'report.json').read_text())
            for seed in plan['scoring_seeds']:
                future,_=read_cells(root/'data/E9.5_RNA.h5ad',panel,2000,seed)
                order=np.random.default_rng(seed).permutation(len(future))
                evaluator=Panel(core,future[order[:1000]],donors,seed)
                floor,ceiling=evaluator.metrics(donors),evaluator.metrics(future[order[1000:]])
                previous=next(v for v in old['panels'] if v['seed']==seed)
                if floor!=previous['floor'] or ceiling!=previous['ceiling']:raise ValueError('Frozen calibration mismatch')
                rows=[]
                for name in plan['candidates']:
                    with cache.read(name,consume=False) as pred: raw=evaluator.metrics(pred)
                    row={'candidate':name,'raw_metrics':raw,**evaluator.aggregate(raw,floor,ceiling)}
                    rows.append(row);emit('candidate_scored',seed=seed,**row)
                report['panels'].append({'seed':seed,'floor':floor,'ceiling':ceiling,'results':rows})
                (RUN/'report.partial.json').write_text(json.dumps(report,indent=2))
            summary=[]
            for name in plan['candidates']:
                rows=[next(v for v in panel_['results'] if v['candidate']==name) for panel_ in report['panels']]
                summary.append({'candidate':name,'scores':[v['local_score'] for v in rows], 'mean_score':float(np.mean([v['local_score'] for v in rows])), 'skills':[v['skills'] for v in rows], 'raw_metrics':[v['raw_metrics'] for v in rows], 'all_calibrations_valid':all(v['calibration_valid'] for v in rows), 'mean_skills':{m:float(np.mean([v['skills'][m] for v in rows])) for m in rows[0]['skills']}})
            by={v['candidate']:v for v in summary}
            passing=[n for n in ['linear_fullgene','neural_fullgene'] if by[n]['all_calibrations_valid'] and all(by[n]['scores'][i]>max(by[c]['scores'][i] for c in ['copy','incumbent']) for i in range(3)) and all(by[n]['mean_skills'][m]>=max(by[c]['mean_skills'][m] for c in ['copy','incumbent']) for m in by[n]['mean_skills'])]
            peak=peak_memory()
            if peak>16*1024**3:raise MemoryError('Host memory ceiling exceeded')
            report.update(status='completed',summary=summary,generation=generation,passing_candidates=passing,official_score=None,local_72_gate_passed=False,scope=plan['scope'],resource_peak_process_working_set_bytes=peak,training={'device':saved['device'],'peak_allocated_bytes':saved['peak_allocated_bytes']})
            (RUN/'report.json').write_text(json.dumps(report,indent=2))
            report['report_sha256']=digest(RUN/'report.json')
            PUBLIC.write_text(json.dumps(report,indent=2))
            emit('completed',report_sha256=report['report_sha256'])
        finally:
            cache.close()
    # Drop only the precisely named, regenerable D-only staging matrix.
    matrix=(RUN/'observed_expression.npy').resolve()
    if matrix.parent!=RUN.resolve() or matrix.drive.upper()!='D:':raise ValueError('Unsafe staging path')
    matrix.unlink()


if __name__=='__main__':
    torch.set_num_threads(2)
    if '--train' in sys.argv: train()
    else: main()
