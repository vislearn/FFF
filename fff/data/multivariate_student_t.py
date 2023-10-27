# adapted from https://github.com/pytorch/pytorch/blob/main/torch/distributions/studentT.py

import math

import torch
from torch.distributions import Chi2, constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _standard_normal


class MultivariateStudentT(Distribution):
    arg_constraints = {'df': constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, df: torch.Tensor, dim: int, validate_args=None):
        # df should be a torch.Tensor of shape (1,)
        assert isinstance(df, torch.Tensor) and len(df.shape) == 1 and df.shape[0] == 1
        self.df = df
        self.dim = dim
        self._chi2 = Chi2(self.df)
        batch_shape = torch.Size((dim,))
        super().__init__(batch_shape, validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        #   X ~ Normal(0, 1)
        #   Z ~ Chi2(df)
        #   Y = X / sqrt(Z / df) ~ StudentT(df)
        shape = self._extended_shape(sample_shape)
        X = _standard_normal(shape, dtype=self.df.dtype, device=self.df.device)
        Z = self._chi2.rsample(sample_shape)
        Y = X * torch.rsqrt(Z / self.df)
        return Y

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        y = value
        dim = len(self.df)
        Z = (0.5 * dim * torch.log(self.df * math.pi) +
             torch.lgamma(0.5 * self.df) -
             torch.lgamma(0.5 * (self.df + dim)))
        # In case self.df is an inference tensor^^
        df = self.df + 0
        return -0.5 * (df + dim) * torch.log1p(torch.sum(y ** 2., -1) / df) - Z
