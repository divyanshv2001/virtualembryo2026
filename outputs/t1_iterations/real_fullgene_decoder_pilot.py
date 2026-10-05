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

TRUST = '--latent-support-trust' in sys.argv
CHANNEL = '--frozen-detection-ridge2' in sys.argv or TRUST
STABILIZE = '--stabilize-ridge2' in sys.argv or CHANNEL
STABILITY = '--head-stability' in sys.argv or STABILIZE
HEAD_RIDGE = 2. if STABILIZE else 1.
HURDLE = '--hurdle' in sys.argv or STABILITY
RUN = HERE / ('private/hurdle_ridge2_stabilization_01' if STABILIZE else ('private/hurdle_head_stability_01' if STABILITY else ('private/observed_fullgene_hurdle_01' if HURDLE else 'private/real_fullgene_decoder_pilot_03')))
PUBLIC = HERE / ('HURDLE_RIDGE2_STABILIZATION_RESULTS.json' if STABILIZE else ('HURDLE_HEAD_STABILITY_RESULTS.json' if STABILITY else ('OBSERVED_FULLGENE_HURDLE_RESULTS.json' if HURDLE else 'REAL_FULLGENE_DECODER_PILOT_RECOVERY_RESULTS.json')))
if CHANNEL:
    RUN=HERE/'private/hurdle_detection_ridge2_01'
    PUBLIC=HERE/'HURDLE_DETECTION_RIDGE2_RESULTS.json'
if TRUST:
    RUN=HERE/'private/hurdle_latent_trust_01'
    PUBLIC=HERE/'HURDLE_LATENT_TRUST_RESULTS.json'


def channel_head_paths():
    return [(label,
             HERE/('private/observed_fullgene_hurdle_01/hurdle.npz' if label=='stabilized_full' else 'private/hurdle_head_stability_01/'+label+'.npz'),
             HERE/('private/hurdle_ridge2_stabilization_01/'+label+'.npz'))
            for label in ['stabilized_full','half0','half1']]


def merge_channel_heads(original, ridge2):
    for key in ['center','centroid','mean','probability','counts']:
        np.testing.assert_array_equal(original[key],ridge2[key])
    merged=dict(original)
    merged['detection']=ridge2['detection'].copy()
    if not all(np.isfinite(value).all() for value in merged.values()):
        raise ValueError('Nonfinite cached channel head')
    return merged


def prepare_channel_heads():
    plan=json.loads((RUN/'plan.json').read_text())
    for filename,expected in plan['cached_head_sha256'].items():
        if digest(Path(filename))!=expected:raise ValueError('Cached channel input changed')
    old=HERE/'private/hurdle_ridge2_stabilization_01'
    packet=json.loads((old/'training.json').read_text())
    if digest(RUN/'coordinates.npy')!=packet['sha256'][str(old/'coordinates.npy')]:
        raise ValueError('Cached training coordinates changed')
    for label,past,new in channel_head_paths():
        model=merge_channel_heads(dict(np.load(past)),dict(np.load(new)))
        np.savez_compressed(RUN/(label+'.npz'),**model)
        append_event(RUN/'events.jsonl','frozen_channel_heads_reused',label=label,
                     positive_sha256=digest(past),detection_sha256=digest(new),fit_stage=8.5)
    (RUN/'hurdle_device.json').write_text(json.dumps({'device':'cached RTX3060 heads; no new training',
        'peak_allocated_bytes':0,'fit_rows':16787,'fit_stage':8.5,
        'solver':'Frozen positive ridge1 + detection ridge2; exact shared fitted sufficient statistics'}))


def hurdle_block(h, response):
    """Match pinned conditional_ridge(ridge=1) with GPU double precision."""
    positive=(response>0).to(h.dtype)
    counts=positive.sum(0); den=counts.clamp_min(1)
    centroid=h.T@positive/den
    means=response.sum(0)/den
    pairs=(h[:,:,None]*h[:,None,:]).reshape(len(h),-1)
    second=(pairs.T@positive/den).T.reshape(len(counts),8,8)
    covariance=second-torch.einsum('ig,jg->gij',centroid,centroid)
    rhs=(h.T@response/den-centroid*means).T
    coefficient=torch.linalg.solve(covariance+HEAD_RIDGE*torch.eye(8,device=h.device,dtype=h.dtype)[None],rhs[:,:,None])[:,:,0].T
    coefficient[:,counts<20]=0
    centered=h-h.mean(0)
    detection=torch.linalg.solve(centered.T@centered/len(h)+HEAD_RIDGE*torch.eye(8,device=h.device,dtype=h.dtype),centered.T@(positive-positive.mean(0))/len(h))
    detection[:,counts<20]=0
    return coefficient,detection,centroid,means,positive.mean(0),counts


def fit_hurdle():
    if not torch.cuda.is_available():raise RuntimeError('RTX3060 unavailable')
    torch.cuda.set_per_process_memory_fraction(.75)
    packet=json.loads((RUN/'training.json').read_text())
    for filename,expected in packet['sha256'].items():
        if digest(Path(filename))!=expected:raise ValueError('Hurdle fit input changed')
    z=np.load(RUN/'coordinates.npy')
    y=np.load(RUN/'observed_expression.npy',mmap_mode='r')
    order=np.random.default_rng(20261005).permutation(len(z))
    groups=[('half0',order[:len(z)//2]),('half1',order[len(z)//2:])] if STABILITY else [('hurdle',np.arange(len(z)))]
    if STABILIZE:groups=[('stabilized_full',np.arange(len(z)))]+groups
    for label,rows in groups:
        center=z[rows].mean(0)
        h=torch.tensor(z[rows].astype(np.float64)-center,device='cuda')
        np.save(RUN/(label+'_fit_rows.npy'),rows)
        outputs=[[],[],[],[],[],[]]
        for start in range(0,32285,256):
            response=torch.tensor(np.asarray(y[:,start:start+256][rows],dtype=np.float64),device='cuda')
            fitted=hurdle_block(h,response)
            if not all(torch.isfinite(v).all() for v in fitted):raise ValueError('Nonfinite hurdle fit')
            for destination,value in zip(outputs,fitted):destination.append(value.cpu().numpy())
            del response,fitted
        arrays=[np.concatenate(v,axis=1 if i<3 else 0) for i,v in enumerate(outputs)]
        np.savez_compressed(RUN/(label+'.npz'),coef=arrays[0],detection=arrays[1],centroid=arrays[2],mean=arrays[3],probability=arrays[4],counts=arrays[5],center=center)
        append_event(RUN/'events.jsonl','cuda_hurdle_heads_fit',label=label,rows=len(rows),fit_stage=8.5)
    (RUN/'hurdle_device.json').write_text(json.dumps({'device':torch.cuda.get_device_name(0),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'fit_rows':len(z),'fit_stage':8.5,'solver':f'CUDA float64 conditionalridge{HEAD_RIDGE:g}/detectionridge{HEAD_RIDGE:g}'}))


def observed_support_step(start, end, past):
    """Bound endpoint radius by max(past99th percentile, donor starting radius)."""
    center=past.mean(0);cov=np.cov(past,rowvar=False)
    ridge=.05*np.trace(cov)/cov.shape[0]
    inverse=np.linalg.inv(cov+ridge*np.eye(cov.shape[0]))
    h=start-center;delta=end-start
    radii=np.einsum('ni,ij,nj->n',past-center,inverse,past-center)
    radius=float(np.quantile(radii,.99))
    r0=np.einsum('ni,ij,nj->n',h,inverse,h)
    bound=np.maximum(radius,r0)
    a=np.einsum('ni,ij,nj->n',delta,inverse,delta)
    b=np.einsum('ni,ij,nj->n',h,inverse,delta)
    c=r0-bound
    root=np.divide(-b+np.sqrt(np.maximum(b*b-a*c,0)),a,out=np.ones_like(a),where=a>1e-14)
    factor=np.clip(root,0,1)
    result=start+factor[:,None]*delta
    final_radius=np.einsum('ni,ij,nj->n',result-center,inverse,result-center)
    if not np.isfinite(result).all() or np.any(final_radius>bound+1e-7):raise ValueError('Observed support bound failed')
    return result,{'quantile':.99,'covariance_ridge_fraction':.05,'training_rows':len(past),
                   'radius_squared':radius,'limited_rows':int((factor<1).sum()),'minimum_step_fraction':float(factor.min()),
                   'mean_step_fraction':float(factor.mean()),'maximum_radius_excess':float(np.max(final_radius-bound))}


def predict_hurdle(donors,z0,z1,fitted,guard,trust=None):
    from neural_hurdle_forecast import systematic_bernoulli
    from robust_population import covariance_change
    start=z0.numpy().astype(float);end=z1.numpy().astype(float)
    trust_audit=None
    if trust is not None:end,trust_audit=observed_support_step(start,end,trust)
    h0=start-fitted['center'];h1=end-fitted['center']
    mass=np.expm1(donors.astype(float)).sum(1)
    attempts=[]
    for backoff in [1.,.5,.25,.125,0.]:
        pred=donors.copy();rng=np.random.default_rng(20260928)
        for start in range(0,32285,512):
            sl=slice(start,min(start+512,32285));original=np.expm1(donors[:,sl].astype(float));positive=original>0
            change=backoff*(h1-h0)@fitted['coef'][:,sl]
            proposed=np.maximum(donors[:,sl]+change,1e-8)
            future=np.expm1(np.minimum(proposed,np.log1p(10000.)))
            ratio=np.divide(future,original,out=np.ones_like(future),where=positive)
            updated=original*np.clip(ratio,.5,2.)
            p0=np.clip(fitted['probability'][sl]+h0@fitted['detection'][:,sl],1e-4,1-1e-4)
            p1=np.clip(fitted['probability'][sl]+h1@fitted['detection'][:,sl],1e-4,1-1e-4)
            dp=np.clip(backoff*(p1-p0),-.25,.25)*(fitted['counts'][sl]>=20)
            added=systematic_bernoulli(np.where((~positive)&(dp>0),np.clip(dp/(1-p0),0,1),0),rng)
            removed=systematic_bernoulli(np.where(positive&(dp<0),np.clip(-dp/p0,0,1),0),rng)
            value=fitted['mean'][sl]+h1@fitted['coef'][:,sl]-(fitted['centroid'][:,sl]*fitted['coef'][:,sl]).sum(0)
            imputed=np.expm1(np.clip(value,1e-8,np.log1p(10000.)))
            updated[added]=imputed[added];updated[removed]=0
            pred[:,sl]=np.log1p(updated).astype(np.float32)
        abundance=np.expm1(pred.astype(float));total=abundance.sum(1)
        erased=(total==0)&(mass>0);abundance[erased]=np.expm1(donors[erased].astype(float));total=abundance.sum(1)
        factor=np.divide(mass,total,out=np.ones_like(mass),where=total>0)
        pred=np.log1p(abundance*factor[:,None]).astype(np.float32)
        covariance=float(covariance_change(donors[:,guard],pred[:,guard]))
        attempts.append({'backoff':backoff,'covariance_change':covariance,**({'observed_support':trust_audit} if trust_audit is not None else {})})
        if np.isfinite(pred).all() and covariance<=.4:
            return pred,attempts
    raise ValueError('Hurdle covariance guard failed')


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
    if HURDLE:
        plan.update(steps=0,candidates=['copy','incumbent','linear_fullgene','hurdle_fullgene'],primary_contrast=['linear_fullgene','hurdle_fullgene'],
                    fit='All16787 real E8.5 cells, all32285genes. CUDA float64 conditionalpositive ridge1 and detectionridge1; support20. No E9.5fit. Fixed frozenCNF/covariance.25 path; bounded systematic detection changes, abundancefactor.5-2, fullmassrestore and existing384gene covariance.4 backoff.',
                    hypothesis='Observed-only full-panel hurdle decoding preserves cell sparsity/co-expression better than the matched dense real-linear decoder; includes5510source-unmapped genes. Established mechanism, distinct observed-fullgene integration; no biologicalbirth claim.')
    if STABILITY:
        plan.update(candidates=['copy','incumbent','fullfit_hurdle','hurdle_half0','hurdle_half1'],
                    primary_contrast=['fullfit_hurdle','hurdle_half0'],
                    split_seed=20261005,training_half_sizes=[8393,8394],
                    hypothesis='Fixed-method training-sample stability confirmation; no new scientific mechanism or tuning.',
                    scope='Two disjoint observed E8.5 cell training halves, historically exposed real E9.5 target. Not independent embryos/targets. Fixed confirmation earns no new-method reward.',
                    fullfit_heads_sha256=digest(HERE/'private/observed_fullgene_hurdle_01/hurdle.npz'))
    if STABILIZE:
        plan.update(candidates=['copy','incumbent','fullfit_hurdle','stabilized_full','hurdle_half0','hurdle_half1'],
                    primary_contrast=['fullfit_hurdle','stabilized_full'],
                    head_ridge=2., support=20,
                    fit='Same observed E8.5 fullgene CUDA analytic hurdle heads; only positive/detection ridge changes1->2 for new fullfit and fixed halves. Original fullfit is immutable exact-replay control.',
                    hypothesis='One fixed stronger ridge reduces training-sample sensitivity while retaining fullfit development gain; user-authorized stabilization, no grid.',
                    stabilization_success='All three new arms beat incumbent aggregate3/3 and have no mean skill regression on any of4metrics; stabilized full mean must also be>=original fullfit mean. Both halves required; no posthoc selection.',
                    scope='Previously exposed E9.5 development stabilization; fixed original disjoint E8.5 halves. Not independent embryos/targets; no readiness promotion.',
                    old_stability_report_sha256=digest(HERE/'HURDLE_HEAD_STABILITY_RESULTS.json'))
    if CHANNEL:
        cached=[p for _,a,b in channel_head_paths() for p in (a,b)]
        plan.update(head_ridge={'positive':1.,'detection':2.},
                    fit='No new training. Reuse original ridge1 positive heads and ridge2 detection heads with exact matching sufficient statistics/coordinates/fullfit or half rows.',
                    hypothesis='Frozen channel attribution: test whether ridge2 detection alone retains direction gains without the combined-ridge MMD loss. One prespecified hybrid, no grid or causal claim.',
                    cached_head_sha256={str(p):digest(p) for p in cached},
                    prior_advisory='Prior cached Jev recommended retain_provisional(.95); human subsequently requests continuation. Preserve prior advice and failed results; this diagnostic has no assumed benefit.')
    if TRUST:
        plan.update(candidates=['copy','incumbent','fullfit_hurdle','stabilized_full','support_trust'],
                    primary_contrast=['stabilized_full','support_trust'],
                    hypothesis='One fixed observed-support ellipsoid clips extrapolated latent steps; same positive1/detection2 heads, detection sampler and mass/covariance guards. No objective grid or hidden-target use.',
                    scope='Exposed realE9.5 development, E8.5-only support fit. Not independent embryos or official E10.5 validation.',
                    support_quantile=.99,support_covariance_ridge_fraction=.05,
                    success='Support-trust beats stabilizedfull aggregate all3panels and no meanfour-skillregression; exactcurrentfullfit replay required. Original gates unchanged.',
                    fixed_channel_report_sha256=digest(HERE/'HURDLE_DETECTION_RIDGE2_RESULTS.json'))
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
        if HURDLE:
            flags=['--stabilize-ridge2'] if STABILIZE else (['--head-stability'] if STABILITY else ['--hurdle'])
            if CHANNEL:prepare_channel_heads()
            else:subprocess.run([str(cuda),'-u',str(Path(__file__))]+flags+['--fit-hurdle'],check=True)
            fitted=dict(np.load(HERE/'private/observed_fullgene_hurdle_01/hurdle.npz' if STABILITY else RUN/'hurdle.npz'))
            device_info=json.loads((RUN/'hurdle_device.json').read_text())
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
            variants=[('linear_fullgene',(z1.numpy()-z0.numpy())@coef)]
            if not HURDLE:variants.append(('neural_fullgene',change.numpy()))
            for name,delta in variants:
                pred=np.maximum(donors+np.clip(delta,-.25,.25),0).astype(np.float32)
                mass=np.expm1(donors.astype(np.float64)).sum(1)
                abundance=np.expm1(pred.astype(np.float64))
                pred=np.log1p(abundance*(mass/np.maximum(abundance.sum(1),1e-12))[:,None]).astype(np.float32)
                if not np.isfinite(pred).all():raise ValueError('Nonfinite forecast')
                generation[name]={'prediction_sha256':cache.put(name,pred)}
                emit('forecast_frozen',candidate=name,**generation[name])
                del pred,abundance
            if HURDLE:
                pred,attempts=predict_hurdle(donors,z0,z1,fitted,np.load(archive/'features.npy'))
                full_name='fullfit_hurdle' if STABILITY else 'hurdle_fullgene'
                generation[full_name]={'prediction_sha256':cache.put(full_name,pred),'guard_attempts':attempts}
                del pred
                old_decoder=json.loads((HERE/'REAL_FULLGENE_DECODER_PILOT_RECOVERY_RESULTS.json').read_text())
                if generation['linear_fullgene']['prediction_sha256']!=old_decoder['generation']['linear_fullgene']['prediction_sha256']:
                    raise ValueError('Dense real-linear exact replay failed')
                if STABILITY:
                    original=json.loads((HERE/'OBSERVED_FULLGENE_HURDLE_RESULTS.json').read_text())
                    if generation[full_name]['prediction_sha256']!=original['generation']['hurdle_fullgene']['prediction_sha256']:
                        raise ValueError('Fullfit hurdle exact replay failed')
                    for label in (['stabilized_full'] if TRUST else (['stabilized_full','half0','half1'] if STABILIZE else ['half0','half1'])):
                        model=dict(np.load(RUN/(label+'.npz')))
                        pred,attempts=predict_hurdle(donors,z0,z1,model,np.load(archive/'features.npy'))
                        name=label if label=='stabilized_full' else 'hurdle_'+label
                        generation[name]={'prediction_sha256':cache.put(name,pred),'guard_attempts':attempts}
                        del pred
            if TRUST:
                archived_channel=json.loads((HERE/'HURDLE_DETECTION_RIDGE2_RESULTS.json').read_text())
                if generation['stabilized_full']['prediction_sha256']!=archived_channel['generation']['stabilized_full']['prediction_sha256']:
                    raise ValueError('Frozen56.761532 control exact replay failed')
                pred,attempts=predict_hurdle(donors,z0,z1,dict(np.load(RUN/'stabilized_full.npz')),np.load(archive/'features.npy'),trust=z.astype(np.float64))
                generation['support_trust']={'prediction_sha256':cache.put('support_trust',pred),'guard_attempts':attempts}
                emit('observed_support_forecast_frozen',**generation['support_trust']);del pred
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
            passing=[n for n in plan['candidates'][2:] if by[n]['all_calibrations_valid'] and all(by[n]['scores'][i]>max(by[c]['scores'][i] for c in ['copy','incumbent']) for i in range(3)) and all(by[n]['mean_skills'][m]>=max(by[c]['mean_skills'][m] for c in ['copy','incumbent']) for m in by[n]['mean_skills'])]
            peak=peak_memory()
            if peak>16*1024**3:raise MemoryError('Host memory ceiling exceeded')
            report.update(status='completed',summary=summary,generation=generation,passing_candidates=passing,official_score=None,local_72_gate_passed=False,scope=plan['scope'],resource_peak_process_working_set_bytes=peak,training=device_info if HURDLE else {'device':saved['device'],'peak_allocated_bytes':saved['peak_allocated_bytes']})
            if TRUST:
                base=by['stabilized_full'];candidate=by['support_trust']
                report['support_trust_gate_passed']=candidate['all_calibrations_valid'] and all(a>b for a,b in zip(candidate['scores'],base['scores'])) and all(candidate['mean_skills'][m]>=base['mean_skills'][m] for m in base['mean_skills'])
            elif STABILIZE:
                arms=['stabilized_full','hurdle_half0','hurdle_half1']
                report['stabilization_gate_passed']=all(n in passing for n in arms) and by['stabilized_full']['mean_score']>=by['fullfit_hurdle']['mean_score']
                report['old_stability_failure_retained']=True
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
    if '--fit-hurdle' in sys.argv: fit_hurdle()
    elif '--train' in sys.argv: train()
    else: main()
