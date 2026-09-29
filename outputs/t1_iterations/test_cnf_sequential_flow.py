import copy
import numpy as np
import torch
from cnf_density_flow import inverse_density,DensityFlowNet
from cnf_sequential_flow import sequential_terms,train_sequential_density


def test_shared_backward_likelihood_and_gradients_match_independent_paths():
    torch.manual_seed(471);samples=[torch.randn(n,3) for n in [3,5,4]];times=[.25,.5,.75]
    coefficient=torch.tensor(.1,requires_grad=True);field=lambda t,z:coefficient*z
    nll,energy=sequential_terms(field,samples,times)
    separate=[inverse_density(field,s.clone(),t) for s,t in zip(samples,times)]
    expected=torch.stack([v[0].mean() for v in separate]).mean();expected_energy=torch.stack([v[1].mean() for v in separate]).mean()
    torch.testing.assert_close(nll,expected);torch.testing.assert_close(energy,expected_energy)
    gradient=torch.autograd.grad(nll+.1*energy,coefficient)[0]
    independent_gradient=torch.autograd.grad(expected+.1*expected_energy,coefficient)[0]
    torch.testing.assert_close(gradient,independent_gradient)


def test_checkpoint_resume_matches_uninterrupted_multistage_training(tmp_path):
    torch.manual_seed(472);net=DensityFlowNet(np.eye(2),np.zeros(2),8.,7.25);other=copy.deepcopy(net)
    rng=np.random.default_rng(472);values=rng.normal(size=(48,2)).astype(np.float32);stages=np.repeat([7.5,7.75,8.],16)
    train_sequential_density(net,values,stages,.1,tmp_path/'short.pt',lambda *a,**k:None,steps=2)
    saved=torch.load(tmp_path/'short.pt',weights_only=False);saved['steps']=4;torch.save(saved,tmp_path/'short.pt')
    train_sequential_density(net,values,stages,.1,tmp_path/'short.pt',lambda *a,**k:None,steps=4,resume=True)
    history=train_sequential_density(other,values,stages,.1,tmp_path/'full.pt',lambda *a,**k:None,steps=4)
    for key,value in net.state_dict().items():torch.testing.assert_close(value,other.state_dict()[key],rtol=0,atol=0)
    assert sum(history[-1]['per_stage_batch_sizes'])==64
    assert history[-1]['all_past_stages']==[7.5,7.75,8.]
