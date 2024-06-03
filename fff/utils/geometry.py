from math import prod
import torch
from geomstats.geometry.manifold import Manifold


def project_jac_to_tangent_space(
    jac: torch.Tensor,
    x_in: torch.Tensor,
    x_out: torch.Tensor,
    manifold: Manifold,
) -> torch.Tensor:
    """Project the Jacobian of a function from x_in to x_out to the tangent
    space of the manifold at x_in and x_out.

    :param jac: Jacobian of the function from x_in to x_out. Shape:
        (batch_size, embedding_dim, embedding_dim).
    :param x_in: Input points. Shape: (batch_size, embedding_dim).
    :param x_out: Output points. Shape: (batch_size, embedding_dim).
    :param manifold: Manifold on which the points lie.
    :return: Projected Jacobian. Shape:
        (batch_size, manifold_dim, manifold_dim)."""

    bases = []
    # Compute a basis each for x, z, and x1
    for pos in [x_in, x_out]:
        bs, dim = pos.shape[0], prod(pos.shape[1:])
        # This is a (bs, dim, manifold_dim) tensor
        tangents = torch.stack([
            random_tangent_vec(manifold, pos, n_samples=bs).reshape(bs, dim)
            for _ in range(manifold.dim)
        ], -1)
        basis, _ = torch.linalg.qr(tangents)
        bases.append(basis)
    x_in_basis, x_out_basis = bases

    # Project the Jacobian after reshaping to bs x out_dim x in_dim
    x_in_dim = prod(x_in.shape[1:])
    x_out_dim = prod(x_out.shape[1:])
    jac_vec = jac.reshape(jac.shape[0], x_out_dim, x_in_dim)
    return torch.bmm(
        torch.bmm(x_out_basis.transpose(-1, -2), jac_vec),
        x_in_basis
    )


def random_tangent_vec(manifold, base_point, n_samples):
    """Generate random tangent vec.

    Copied from geomstats with the right device handling.
    Also do not squeeze the batch dimension.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    base_point :  array-like, shape={[n_samples, *point_shape]}
        Point.

    Returns
    -------
    tangent_vec : array-like, shape=[..., *point_shape]
        Tangent vec at base point.
    """
    if (
            n_samples > 1
            and base_point.ndim > len(manifold.shape)
            and n_samples != len(base_point)
    ):
        raise ValueError(
            "The number of base points must be the same as the "
            "number of samples, when the number of base points is different from 1."
        )
    return manifold.to_tangent(
        torch.randn(size=(n_samples,) + manifold.shape, device=base_point.device), base_point
    )
