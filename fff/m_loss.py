from math import prod, sqrt

import torch
from torch.autograd.forward_ad import unpack_dual

from fff.m_fff import random_tangent_vec


def sample_v_in_tangent(x: torch.Tensor, hutchinson_samples: int, manifold):
    """Sample a vector in the tangent space of x.

    Parameters
    ----------
    x : torch.Tensor
        A point on the manifold.
    hutchinson_samples : int
        Number of samples to use for the Hutchinson trace estimator.
    manifold : geomstats.geometry.manifold.Manifold
        The manifold on which x lies.

    Returns
    -------
    v : torch.Tensor
        A vector in the tangent space of x.
    """
    batch_size, total_dim = x.shape[0], prod(x.shape[1:])
    if hutchinson_samples != 1:
        raise NotImplementedError(f"More than one Hutchinson sample not implemented for M-FFF, "
                                  f"{hutchinson_samples} requested.")
    v = random_tangent_vec(manifold, x.detach(), n_samples=batch_size)
    v = v / torch.norm(v, p=2, dim=list(range(1, len(v.shape))), keepdim=True)
    return v[..., None] * sqrt(total_dim)


def reconstruction_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sum((a - b) ** 2, dim=tuple(range(1, len(a.shape))))


def project_z(z, manifold):
    z_projected = manifold.projection(z)
    z_projection_distance = reconstruction_loss(z, z_projected)
    return z_projected, z_projection_distance


def project_x1(dual_x1, manifold):
    dual_x1_projected = manifold.projection(dual_x1)
    x1_projection_distance = reconstruction_loss(
        unpack_dual(dual_x1_projected)[0], unpack_dual(dual_x1)[0]
    )
    return dual_x1_projected, x1_projection_distance
