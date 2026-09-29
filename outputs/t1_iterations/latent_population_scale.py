"""Predeclared latent population contraction/expansion sensitivity."""
import torch


class LatentPopulationScale:
    def __init__(self, base, factor=1.):
        self.base = base
        self.cutoff = base.cutoff
        self.origin = base.origin
        self.configure(factor)

    def configure(self, factor):
        if factor not in [.75, 1., 1.25]:
            raise ValueError('Undeclared population scale')
        self.factor = factor

    def encode(self, values):
        return self.base.encode(values)

    def trajectory(self, z, times):
        result = self.base.trajectory(z, times)
        if self.factor == 1.:
            return result
        result = result.clone()
        for i in range(1, len(times)):
            elapsed = float(times[i] - times[0])
            if elapsed < 0:
                raise ValueError('Backward population scaling undeclared')
            # A sensitivity per stage unit, not a learned biological rate.
            scale = self.factor ** elapsed
            center = result[i].mean(0, keepdim=True)
            result[i] = center + scale * (result[i] - center)
        return result
