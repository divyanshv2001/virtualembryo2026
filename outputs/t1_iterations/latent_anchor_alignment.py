"""Past-anchor mean/covariance alignment for prediction-time encoding only."""
import numpy as np
import torch
from torch import nn


def covariance_map(source,anchor,ridge=.1):
    if ridge not in [0.,.1] or min(len(source),len(anchor))<2:raise ValueError('Invalid covariance alignment')
    def root(values,power):
        covariance=np.cov(values,rowvar=False)+ridge*np.eye(values.shape[1]);eigen,basis=np.linalg.eigh(covariance)
        if eigen.min()<=0:raise ValueError('Singular covariance')
        return (basis*eigen**power)@basis.T
    matrix=root(anchor,-.5)@root(source,.5)
    if not np.isfinite(matrix).all():raise ValueError('Invalid alignment matrix')
    return matrix


class LatentAlignedNet(nn.Module):
    def __init__(self,base,source,anchor,mode='covariance',strength=1.):
        super().__init__();self.base=base
        if mode not in ['mean','covariance'] or strength not in [0.,.25,.5,1.]:raise ValueError('Undeclared alignment')
        self.strength=strength;self.mode=mode
        matrix=covariance_map(source,anchor) if mode=='covariance' else np.eye(source.shape[1])
        self.register_buffer('matrix',torch.tensor(matrix,dtype=torch.float32))
        self.register_buffer('anchor_mean',torch.tensor(anchor.mean(0),dtype=torch.float32))
        self.register_buffer('source_mean',torch.tensor(source.mean(0),dtype=torch.float32))
    def encode(self,values):
        z,aux=self.base.encode(values)
        if self.strength==0:return z,aux
        transformed=(z-self.anchor_mean)@self.matrix+self.source_mean
        return z+self.strength*(transformed-z),aux
    def trajectory(self,z,times):return self.base.trajectory(z,times)


def predict_latent_aligned(program,target,source,mode,strength,forecast_mode):
    original=program.net
    with torch.no_grad():
        source_z=original.encode(torch.tensor((source-program.center)/program.scale))[0].numpy()
        anchor_z=original.encode(torch.tensor((program.donors[:,program.features]-program.center)/program.scale))[0].numpy()
    wrapper=LatentAlignedNet(original,source_z,anchor_z,mode,strength)
    program.net=wrapper
    try:prediction,indices,audit=program.predict(target,forecast_mode,1.,sampling='systematic')
    finally:program.net=original
    audit.update(alignment_mode=mode,alignment_strength=strength,alignment_ridge=.1,
        alignment_source_cells=len(source_z),alignment_anchor_cells=len(anchor_z),
        alignment_latent_mean_gap=float(np.linalg.norm(source_z.mean(0)-anchor_z.mean(0))),
        alignment_matrix_singular_values=np.linalg.svd(wrapper.matrix.numpy(),compute_uv=False).tolist(),
        alignment_scope='Only forecast-time donor encoding. Current-stage atlas and challenge past anchors; no future expression or head/dynamics refit. Reverse-direction CORAL-inspired adaptation, not source classifier retraining. Alignment can erase genuine population shifts.')
    return prediction,indices,audit
