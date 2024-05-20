from torch.distributions import Distribution, Uniform
from torch.distributions.constraints import Constraint
import torch
import numpy as np


class EnergyDistribution(Distribution):

    def __init__(
        self,
        loc: torch.Tensor,
        scale: float,
        rotation: float,
        Z: float,
        *args,
        **kwargs
    ):
        self.loc = loc
        self.scale = scale
        self.Z = Z
        alpha = torch.tensor(rotation)
        self.R = torch.tensor(
            [
                [torch.cos(alpha), -torch.sin(alpha)],
                [torch.sin(alpha), torch.cos(alpha)],
            ]
        )
        low = torch.tensor([-4.0, -3.0])
        high = torch.tensor([4.0, 10.0])
        self.base_distribution = Uniform(low, high)
        super().__init__(*args, **kwargs)

    def energy_fn(self, x):
        x = x
        x1 = x[..., 0]
        x2 = x[..., 1]
        energy = torch.exp(-(((x2 - x1**2) ** 2) + 0.08 * x2**2) / self.scale)
        energy = torch.where(energy > 1e-3, energy, torch.zeros_like(energy))
        return energy

    def sample(self, shape):
        shape_base = (shape[0] * 100, *shape[1:])
        samples = self.base_distribution.sample(shape_base)
        energy = self.energy_fn(samples)
        indices = np.random.choice(
            np.arange(shape_base[0]),
            shape[0],
            p=(energy / energy.sum()).detach().numpy(),
        )
        return (self.R @ samples[indices].T).T + self.loc

    def log_prob(self, value):
        value = (self.R.T @ (value - self.loc).T).T
        return torch.log(self.energy_fn(value) / self.Z)

    @property
    def support(self):
        return EnergyConstraint(self.energy_fn)


class EnergyConstraint(Constraint):
    def __init__(self, energy_fn, *args, **kwargs):
        self.energy_fn = energy_fn
        super().__init__(*args, **kwargs)

    def check(self, value):
        return (self.energy_fn(value) > 0.0).unsqueeze(-1)
