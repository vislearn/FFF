from typing import Optional, Type

import torch

from abc import ABC, abstractmethod
from torch.distributions import (
    Distribution,
    Uniform,
    Normal,
    Categorical,
)
from geomstats.geometry.manifold import Manifold

from fff.data.manifold import fix_device
from fff.utils.manifolds import PoincareBall_, Hyperboloid_
from fff.distributions.mixture_distribution import MixtureDistribution
from fff.distributions.energy_toy_distribution import EnergyDistribution


class WrappedAtOriginDistribution(Distribution, ABC):
    arg_constraints = {}

    def __init__(
        self,
        tangent_distribution: Distribution,
        manifold: Manifold,
        *args,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        self.tangent_distribution = tangent_distribution
        self.manifold = manifold
        self.metric = manifold.default_metric()(manifold)
        self.device = device if device is not None else torch.device("cpu")
        super().__init__(*args, event_shape=manifold.shape, **kwargs)

    @property
    @abstractmethod
    def origin(self) -> torch.Tensor:
        """Defines the origin of the manifold."""
        pass

    @abstractmethod
    def to_tangent_at_origin(self, v: torch.Tensor) -> torch.Tensor:
        """Transforms a d dimensional tangent vector to a D dimensional embedded tangent
        vector in the tangent space at the origin.
        :param v: Tensor of shape (..., d)
        :return: Tensor of shape (..., D)"""
        pass

    @abstractmethod
    def from_tangent_at_origin(self, v: torch.Tensor) -> torch.Tensor:
        """Transforms a D dimensional embedded tangent vector in the tangent space at the
        origin to a d dimensional tangent vector.
        :param v: Tensor of shape (..., D)
        :return: Tensor of shape (..., d)"""
        pass

    @abstractmethod
    def volume_change(self, v: torch.Tensor) -> torch.Tensor:
        """Computes the volume change induced by the exponential map at the origin and the
        change in metric from tangent space to manifold.
        :param v: Tensor of shape (..., D)"""
        pass

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        tangent_samples = self.tangent_distribution.sample(sample_shape)
        tangent_samples = self.to_tangent_at_origin(tangent_samples)
        samples = fix_device(self.metric.exp)(tangent_samples, base_point=self.origin)
        return samples

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        tangent_samples = fix_device(self.metric.log)(value, base_point=self.origin)
        volume_change = self.volume_change(tangent_samples)
        tangent_samples = self.from_tangent_at_origin(tangent_samples)
        tangent_log_probs = self.tangent_distribution.log_prob(tangent_samples)
        return tangent_log_probs.sum(dim=-1) + volume_change


class PoincareDiskWrappedDistribution(WrappedAtOriginDistribution):
    def __init__(
        self,
        tangent_distribution,
        *args,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__(
            tangent_distribution, PoincareBall_(2), *args, device=device, **kwargs
        )

    @property
    def origin(self) -> torch.Tensor:
        return torch.zeros(2, device=self.device)

    def to_tangent_at_origin(self, v: torch.Tensor) -> torch.Tensor:
        return v

    def from_tangent_at_origin(self, v: torch.Tensor) -> torch.Tensor:
        return v

    def volume_change(self, v: torch.Tensor) -> torch.Tensor:
        point, volume_change_exp = self.metric.exp0_with_jac_log_det(v)
        vol_change_metric = self.metric.metric_matrix_log_det(point)
        return - volume_change_exp - 0.5 * vol_change_metric


class HyperboloidWrappedDistribution(WrappedAtOriginDistribution):
    def __init__(self, tangent_distribution, *args, device=None, **kwargs):
        super().__init__(
            tangent_distribution, Hyperboloid_(2), *args, device=device, **kwargs
        )

    @property
    def origin(self) -> torch.Tensor:
        return torch.tensor([1.0, 0.0, 0.0], device=self.device)

    def to_tangent_at_origin(self, v: torch.Tensor) -> torch.Tensor:
        v = 2 * v  # Pushforward to the hyperboloid tangent space at the origin
        return torch.cat([torch.zeros_like(v[:, :1]), v], dim=-1)

    def from_tangent_at_origin(self, v: torch.Tensor) -> torch.Tensor:
        v = 0.5 * v  # Pullback to the PoincareBall tangent space at the origin
        return v[:, 1:]

    def volume_change(self, v: torch.Tensor) -> torch.Tensor:
        sq_norm = self.manifold.embedding_space.metric.squared_norm(v)
        n = self.manifold.dim
        log_sinh = torch.log(torch.sinh(sq_norm**0.5))
        log_norm = 0.5 * torch.log(sq_norm)
        volume_change_exp = (1 - n) * (log_sinh - log_norm)
        volume_change_metric = 0
        return volume_change_exp + volume_change_metric


def get_wrapped_normal_distribution(
    manifold: Manifold,
    *args,
    tangent_scale: float = 1.0,
    device: Optional[torch.device] = None,
    **kwargs,
):
    tangent_distribution = Normal(
        torch.zeros(manifold.dim, device=device),
        torch.ones(manifold.dim, device=device) * tangent_scale,
    )

    if isinstance(manifold, PoincareBall_):
        return PoincareDiskWrappedDistribution(
            tangent_distribution, *args, device=device, **kwargs
        )
    elif isinstance(manifold, Hyperboloid_):
        return HyperboloidWrappedDistribution(
            tangent_distribution, *args, device=device, **kwargs
        )
    else:
        raise NotImplementedError("Manifold not implemented")


def get_checkerboard_distribution(
    side_length: float = 0.75,
    wrapped_distribution: Type[Distribution] = PoincareDiskWrappedDistribution,
) -> Distribution:
    offsets = (
        torch.tensor(
            [
                [0.0, 0.0],
                [0.0, -2.0],
                [1.0, 1.0],
                [1.0, -1.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
                [-2.0, 0.0],
                [-2.0, -2.0],
            ]
        )
        * side_length
    )

    num_distr = len(offsets)
    tangent_distributions = [
        Uniform(
            torch.zeros(2) + offsets[i],
            torch.ones(2) * side_length + offsets[i],
        )
        for i in range(num_distr)
    ]

    mix = Categorical(torch.ones(num_distr) / num_distr)

    tangent_distribution = MixtureDistribution(
        mix, tangent_distributions, finite_support=True
    )

    return wrapped_distribution(tangent_distribution)


def get_five_gaussians_distribution(
    radius: float = 1.5,
    wrapped_distribution: Type[Distribution] = PoincareDiskWrappedDistribution,
) -> Distribution:
    centers = (
        torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.0, 0.0]])
        * radius
    )

    components = [
        wrapped_distribution(Normal(torch.zeros(2) + centers[i], torch.ones(2) / 4))
        for i in range(len(centers))
    ]
    mix = Categorical(torch.ones(len(centers)) / len(centers))

    return MixtureDistribution(mix, components, finite_support=False)


def get_swish_distribution(
    s: float = 0.7,
    var: tuple = (0.2, 0.6),
    wrapped_distribution: Type[Distribution] = PoincareDiskWrappedDistribution,
) -> Distribution:
    locs = torch.tensor(
        [
            [s, s],
            [-s, -s],
            [-s, s],
            [s, -s],
        ]
    )

    scales = torch.tensor(
        [[var[0], var[1]], [var[0], var[1]], [var[1], var[0]], [var[1], var[0]]]
    )

    tangent_distributions = [Normal(locs[i], scales[i]) for i in range(4)]
    components = [
        wrapped_distribution(tangent_distributions[i], locs[i]) for i in range(4)
    ]
    mix = Categorical(torch.ones(4) / 4)

    return MixtureDistribution(mix, components, finite_support=False)


def get_swish_distribution_v2(
    wrapped_distribution: Type[Distribution] = PoincareDiskWrappedDistribution,
):
    locs = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]]) * 0.6
    rotation_offset = -torch.pi / 5
    rotations = torch.tensor(
        [
            rotation_offset - torch.pi / 2,
            rotation_offset,
            rotation_offset + torch.pi / 2,
            rotation_offset + torch.pi,
        ]
    )
    tangent_distributions = [
        EnergyDistribution(
            loc=locs[i], scale=0.01, rotation=rotations[i], Z=0.14377858780888922
        )
        for i in range(4)
    ]

    mix = Categorical(torch.ones(4) / 4)

    return wrapped_distribution(
        MixtureDistribution(mix, tangent_distributions, finite_support=True)
    )


def get_one_wrapped_distribution(
    wrapped_distribution: Type[Distribution] = PoincareDiskWrappedDistribution,
) -> Distribution:
    scale = torch.tensor([0.3, 0.3])

    tangent_distribution = Normal(torch.tensor([-0.6, 0.6]), scale)

    return wrapped_distribution(tangent_distribution)
