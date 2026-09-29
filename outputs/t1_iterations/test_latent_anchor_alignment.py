import numpy as np
import torch
from cnf_density_flow import DensityFlowNet
from latent_anchor_alignment import covariance_map,LatentAlignedNet


def test_whitening_coloring_matches_full_covariance_without_ridge():
    rng=np.random.default_rng(521);anchor=rng.normal(size=(300,3));source=rng.normal(size=(300,3))@np.array([[2.,.1,0],[.2,1.,.3],[0,.2,.7]])+3
    matrix=covariance_map(source,anchor,0.)
    mapped=(anchor-anchor.mean(0))@matrix+source.mean(0)
    np.testing.assert_allclose(np.cov(mapped,rowvar=False),np.cov(source,rowvar=False),atol=1e-12)
    np.testing.assert_allclose(mapped.mean(0),source.mean(0),atol=1e-12)


def test_zero_alignment_is_bit_exact_and_mean_alignment_preserves_covariance():
    rng=np.random.default_rng(522);anchor=rng.normal(size=(40,3)).astype(np.float32);source=anchor+1
    torch.manual_seed(522);base=DensityFlowNet(np.eye(3),np.zeros(3),8.5,7.25)
    tensor=torch.tensor(anchor);zero=LatentAlignedNet(base,source,anchor,'covariance',0.)
    torch.testing.assert_close(zero.encode(tensor)[0],base.encode(tensor)[0],rtol=0,atol=0)
    mean=LatentAlignedNet(base,source,anchor,'mean',1.).encode(tensor)[0].detach().numpy()
    np.testing.assert_allclose(np.cov(mean,rowvar=False),np.cov(anchor,rowvar=False),atol=1e-7)
