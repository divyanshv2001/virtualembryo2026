"""Past-only, full E8.5 anchor slope fit with bounded memory.

The forecast still starts from the same 1,500 observed donor cells. The
additional E8.5 cells only estimate conditional expression/detection slopes.
"""
import anndata as ad
import numpy as np
import torch
from scipy import sparse

from partial_anchor_forecast import PartialAnchorForecast
from ridge_conditional_head import conditional_ridge


def fit_anchor_block(h, response, source_positive_coef, source_detection_coef,
                     source_support, ridge=1.):
    """Fit a gene block; return deltas, centers, counts and supported flags."""
    if response.shape[0] != len(h):
        raise ValueError('Anchor expression/latent row mismatch')
    positive = (response > 0).astype(float)
    count = positive.sum(0)
    supported = (count >= 20) & source_support
    conditional = conditional_ridge(h, positive, response * positive, ridge=ridge)
    hcenter = h.mean(0)
    centered = h - hcenter
    gram = centered.T @ centered / len(h) + ridge * np.eye(h.shape[1])
    detection = np.linalg.solve(gram, centered.T @ (positive - positive.mean(0)) / len(h))
    positive_center = h.T @ positive / np.maximum(count, 1)
    return ((conditional - source_positive_coef) * supported,
            (detection - source_detection_coef) * supported,
            positive_center, count.astype(int), supported)


class FullAnchorSlopeForecast(PartialAnchorForecast):
    """Replace donor-sampled calibration slopes using all permitted E8.5 cells."""

    def __init__(self, *args, anchor_path, **kwargs):
        super().__init__(*args, **kwargs)
        panel = args[4]
        a = ad.read_h5ad(anchor_path, backed='r')
        try:
            if not a.var_names.is_unique or a.var_names.tolist() != panel:
                raise ValueError('Full-anchor panel mismatch')
            n = a.n_obs
            h = np.empty((n, len(self.zcenter)), dtype=np.float64)
            # Row blocks cap the feature/encoder tensor even with 16 GB RAM.
            with torch.no_grad():
                for start in range(0, n, 256):
                    end = min(start + 256, n)
                    block = a.X[start:end, self.features]
                    block = block.toarray() if sparse.issparse(block) else np.asarray(block)
                    z = self.net.encode(torch.tensor((block.astype(np.float32) - self.center) / self.scale))[0]
                    h[start:end] = z.numpy().astype(float) - self.zcenter
            if not np.isfinite(h).all():
                raise ValueError('Nonfinite full-anchor coordinates')
            self.anchor_latent_mean = h.mean(0)
            for start in range(0, len(self.mapped), 256):
                sl = slice(start, min(start + 256, len(self.mapped)))
                block = a.X[:, self.mapped[sl]]
                response = block.toarray() if sparse.issparse(block) else np.asarray(block)
                if not np.isfinite(response).all() or (response < 0).any():
                    raise ValueError('Invalid full-anchor expression')
                (self.anchor_positive_delta[:, sl], self.anchor_detection_delta[:, sl],
                 self.anchor_positive_center[:, sl], self.anchor_count[sl],
                 self.calibration_support[sl]) = fit_anchor_block(
                    h, response.astype(np.float64), self.positive_coef[:, sl],
                    self.detection[:, sl], self.support[sl])
            self.audit.update(anchor_calibration_rows=n,
                              anchor_supported_genes=int(self.calibration_support.sum()),
                              method='All-past-E8.5 conditional anchor slopes, source levels fixed',
                              limitation='Cross-sectional E8.5 associations are not temporal velocities.')
            self.configure(0., 0.)
        finally:
            a.file.close()
