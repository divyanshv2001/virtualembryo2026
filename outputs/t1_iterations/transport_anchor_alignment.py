"""Anchor-only moment alignment used exclusively during forecast encoding."""
import numpy as np
import torch
from torch import nn


class AnchorAlignedNet(nn.Module):
    def __init__(self,net,source,anchor,center,scale):
        super().__init__();self.base=net
        self.register_buffer('anchor_center',torch.tensor((anchor.mean(0)-center)/scale,dtype=torch.float32))
        self.register_buffer('source_center',torch.tensor((source.mean(0)-center)/scale,dtype=torch.float32))
        self.register_buffer('ratio',torch.tensor(np.clip(source.std(0)/np.maximum(anchor.std(0),.1),.5,2.),dtype=torch.float32))
    def encode(self,values):
        return self.base.encode((values-self.anchor_center)*self.ratio+self.source_center)
    def trajectory(self,z,times):return self.base.trajectory(z,times)


def predict_aligned(model,target,source,alignment):
    if alignment not in ['identity','moments']:raise ValueError('Unknown alignment')
    original=model.net
    if alignment=='moments':model.net=AnchorAlignedNet(original,source,model.donors[:,model.features],model.center,model.scale)
    try:pred,indices,audit=model.predict(target,'abundance',.5)
    finally:model.net=original
    audit.update(alignment=alignment,alignment_scope='Only prediction-time donor encoding; source-fitted heads and transport unchanged. Source current stage and challenge past anchors only. Moment matching is a sensitivity test, not validated domain correction.')
    return pred,indices,audit
