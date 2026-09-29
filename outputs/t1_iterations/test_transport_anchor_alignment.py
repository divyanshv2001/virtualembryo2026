import numpy as np
import torch
from transport_anchor_alignment import AnchorAlignedNet,predict_aligned
from transport_latent_flow import AffineTransportNet


def test_anchor_alignment_matches_explicit_past_only_feature_transform():
    rng=np.random.default_rng(440);source=rng.normal(size=(40,5)).astype(np.float32)
    anchor=(rng.normal(size=(30,5))*2+1).astype(np.float32)
    center=np.full(5,.1,dtype=np.float32);scale=np.full(5,1.2,dtype=np.float32)
    base=AffineTransportNet(np.eye(5)[:3],np.zeros(5),np.zeros((4,3)))
    net=AnchorAlignedNet(base,source,anchor,center,scale)
    ratio=np.clip(source.std(0)/np.maximum(anchor.std(0),.1),.5,2.)
    expected=((anchor-anchor.mean(0))*ratio+source.mean(0)-center)/scale
    got=net.encode(torch.tensor((anchor-center)/scale))[0]
    np.testing.assert_allclose(got.numpy(),base.encode(torch.tensor(expected))[0].numpy(),atol=3e-7)
    same=AnchorAlignedNet(base,source,source,center,scale)
    normalized=torch.tensor((source-center)/scale)
    np.testing.assert_allclose(same.encode(normalized)[0].numpy(),base.encode(normalized)[0].numpy(),atol=3e-7)
    class FailingModel:
        net=base;features=np.arange(5);donors=anchor
        def predict(self,*args):raise RuntimeError('projection failed')
    model=FailingModel();model.center=center;model.scale=scale
    try:predict_aligned(model,9.5,source,'moments')
    except RuntimeError:pass
    assert model.net is base
