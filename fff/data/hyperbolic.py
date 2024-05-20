import torch
from torch.utils.data import TensorDataset
import numpy as np

from math import pi

from .utils import split_dataset
from .manifold import ManifoldDataset

from fff.utils.manifolds import PoincareBall_
from fff.distributions.wrapped_at_origin import (
    get_checkerboard_distribution,
    get_five_gaussians_distribution,
    get_swish_distribution,
    get_swish_distribution_v2,
    get_one_wrapped_distribution,
)


def get_hyperbolic_toy_dataset(subtype: str, seed: int = np.random.seed()):
    if subtype == "checkerboard":
        distribution = get_checkerboard_distribution()
    elif subtype == "five_gaussians":
        distribution = get_five_gaussians_distribution()
    elif subtype == "swish":
        distribution = get_swish_distribution()
    elif subtype == "swish_v2":
        distribution = get_swish_distribution_v2()
    elif subtype == "one_wrapped":
        distribution = get_one_wrapped_distribution()
    else:
        raise ValueError(f"Unknown subtype: {subtype}")

    data = distribution.sample((100_000,))
    train_data, val_data, test_data = split_dataset(data, seed=seed)
    manifold = PoincareBall_(dim=2)
    return (
        ManifoldDataset(TensorDataset(train_data), manifold),
        ManifoldDataset(TensorDataset(val_data), manifold),
        ManifoldDataset(TensorDataset(test_data), manifold),
    )


def get_hyperbolic_wrapped_normal_dataset(
    dim: int,
    size: int = 100000,
    seed: int = np.random.seed(),
    tangent_scale: float = 0.1,
    radius: float = 0.7,
):
    manifold = PoincareBall_(dim=dim)
    metric = manifold.default_metric()(manifold)
    # distribution = WrappedStandardNormal(manifold, tangent_scale=tangent_scale)
    # data = distribution.rsample((size,))

    base_points = torch.stack(
        [
            radius * torch.tensor([np.cos(angle), np.sin(angle)], dtype=torch.float32)
            for angle in [0, pi, pi / 2, 3 * pi / 2]
        ]
    )

    data = []
    for base_point in base_points:
        tangent = tangent_scale * manifold.random_tangent_vec(
            base_point, size // len(base_points)
        )
        data.append(metric.exp(tangent, base_point))

    data = torch.cat(data)
    data = data[torch.randperm(data.shape[0]), :]

    train_data, val_data, test_data = split_dataset(data, seed=seed)

    return (
        ManifoldDataset(TensorDataset(train_data), manifold),
        ManifoldDataset(TensorDataset(val_data), manifold),
        ManifoldDataset(TensorDataset(test_data), manifold),
    )


def get_mcnf_data(subtype: str, seed: int = np.random.seed()):
    """Gets the data introduced in the paper 'Neural Manifold Ordinary Differential Equations'
    https://arxiv.org/abs/2006.10254. The data is generated using the code provided in the repository
    https://github.com/CUAI/Neural-Manifold-Ordinary-Differential-Equations/tree/master and mapped to
    the poincare ball model."""

    data = torch.load(f"fff/data/raw_data/hyperbolic/{subtype}.pt")
    train_data, val_data, test_data = split_dataset(data, seed=seed)
    manifold = PoincareBall_(dim=2)

    return (
        ManifoldDataset(TensorDataset(train_data), manifold),
        ManifoldDataset(TensorDataset(val_data), manifold),
        ManifoldDataset(TensorDataset(test_data), manifold),
    )
