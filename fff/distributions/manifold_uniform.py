import math
import warnings

import numpy as np
import torch
from torch.distributions import Distribution


class ManifoldUniformDistribution(Distribution):
    arg_constraints = {}

    def __init__(self, manifold, *args, device=None, **kwargs):
        super().__init__(*args, event_shape=manifold.shape, **kwargs)
        self.manifold = manifold
        self.dim = manifold.dim

        from geomstats.geometry.hypersphere import Hypersphere
        from geomstats.geometry.product_manifold import ProductManifold
        from geomstats.geometry.special_orthogonal import _SpecialOrthogonalMatrices
        if isinstance(manifold, Hypersphere):
            # print(f"Setting normalization constant for {n}-hypersphere")
            n = manifold.dim
            normalization_constant = 2 * (np.pi ** ((n + 1) / 2)) / np.math.gamma((n + 1) / 2)
        elif isinstance(manifold, ProductManifold):
            # Only works for tori
            assert all(
                isinstance(factor, Hypersphere) and factor.dim == 1
                for factor in manifold.factors
            )
            normalization_constant = (2 * np.pi) ** manifold.dim
        elif isinstance(manifold, _SpecialOrthogonalMatrices):
            """https://arxiv.org/pdf/math-ph/0210033.pdf"""
            n = manifold.n
            if n == 2:
                normalization_constant = 2 * math.pi
            elif n == 3:
                normalization_constant = 8 * math.pi ** 2
            else:
                out = (self.n - 1) * math.log(2)
                out += ((self.n - 1) * (self.n + 2) / 4) * math.log(math.pi)
                k = torch.expand_dims(torch.arange(2, self.n + 1), axis=-1)
                out += torch.sum(np.lgamma(k / 2), axis=0)
                normalization_constant = math.exp(out)
        else:
            warnings.warn(f"No normalization constant could be set for manifold {manifold}, "
                          f"the distribution will be unnormalized.")
            normalization_constant = 1

        self.log_norm = - np.log(normalization_constant)
        self.device = device

    def rsample(self, sample_shape=torch.Size()):
        from geomstats.geometry.product_manifold import ProductManifold
        from geomstats.geometry.special_orthogonal import _SpecialOrthogonalMatrices
        if isinstance(self.manifold, ProductManifold):
            # Hack because geomstats does not have random_uniform for product manifold
            samples = self.manifold.random_point(n_samples=math.prod(sample_shape))
        elif isinstance(self.manifold, _SpecialOrthogonalMatrices):
            # Use scipy to sample (geomstats has other distribution)
            from scipy.stats import special_ortho_group
            scipy_samples = special_ortho_group(self.manifold.n).rvs(size=math.prod(sample_shape))
            samples = torch.from_numpy(scipy_samples).float().to(self.device)
        else:
            samples = self.manifold.random_uniform(n_samples=math.prod(sample_shape))
        return samples.reshape([*sample_shape, *self.event_shape]).to(self.device)

    def log_prob(self, value):
        return self.log_norm * torch.ones(value.shape[0], device=self.device)
