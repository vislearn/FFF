from torch.utils.data import TensorDataset
import numpy as np

from .utils import split_dataset
from .manifold import ManifoldDataset

from fff.utils.manifolds import PoincareBall_
from fff.distributions.wrapped_at_origin import (
    get_checkerboard_distribution,
    get_five_gaussians_distribution,
    get_swish_distribution,
    get_one_gaussian_distribution,
)


def get_hyperbolic_toy_dataset(subtype: str, seed: int = np.random.seed()):
    if subtype == "checkerboard":
        distribution = get_checkerboard_distribution()
    elif subtype == "five_gaussians":
        distribution = get_five_gaussians_distribution()
    elif subtype in ["swish", "swish_v2"]:  # swish_v2 only for old experiments
        distribution = get_swish_distribution()
    elif subtype in ["one_gaussian", "one_wrapped"]:  # one_wrapped only for old experiments
        distribution = get_one_gaussian_distribution()
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
