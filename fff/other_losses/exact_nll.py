from collections import namedtuple
from geomstats.geometry.manifold import Manifold

import torch
from torch.distributions import Distribution

from fff.loss import reconstruction_loss, volume_change_metric
from fff.utils.geometry import project_jac_to_tangent_space
from fff.utils.func import compute_jacobian, compute_volume_change
from fff.utils.types import Transform


ExactOutput = namedtuple("ExactOutput", ["exact", "z", "x1", "regularizations"])
NLLExactOutput = namedtuple("NLLExactOutput", ["nll", "z", "x1", "regularizations"])


def volume_change_exact(
    x: torch.Tensor,
    encode: Transform,
    decode: Transform,
    manifold: Manifold | None = None,
) -> ExactOutput:
    """Computes the exact volume change term in the change of variables
    formula. Uses the negative volume change induced by the decode function.

    :param x: Input data. Shape: (batch_size, ...)
    :param encode: Encoder function. Takes `x` as input and returns a latent
        representation `z` of shape (batch_size, latent_shape).
    :param decode: Decoder function. Takes a latent representation `z` as input
        and returns a reconstruction `x1`.
    :param manifold: Optional manifold. If provided the outputs of encode and
        decode are projected to the manifold and the volume change is computed
        in the tangent space of the manifold.
    :return: The exact volume change term of shape (batch_size,), the latent
        representation `z`, the reconstruction `x1` and the regularizations
        computed on the fly.
    """

    regularizations = {}
    exact = 0

    x.requires_grad_()
    z = encode(x)

    # M-FFF: Project to manifold and store projection distance for
    # regularization
    if manifold is not None:
        z_projected = manifold.projection(z)
        regularizations["z_projection"] = reconstruction_loss(z, z_projected)
        z = z_projected

    x1, jac_dec = compute_jacobian(z, decode)

    # M-FFF: Project to manifold and store projection distance for
    # regularization. Also compute the jacobian of the projection and
    # project the jacobian to the tangent spaces of z and x1.
    if manifold is not None:
        x1_projected, jac_proj = compute_jacobian(x1, manifold.projection)
        regularizations["x1_projection"] = reconstruction_loss(x1, x1_projected)
        x1 = x1_projected
        jac_dec = jac_proj @ jac_dec
        jac_dec = project_jac_to_tangent_space(jac_dec, z, x1, manifold)

    exact = -compute_volume_change(jac_dec)

    return ExactOutput(exact, z, x1, regularizations)


def exact_nll(
    x: torch.Tensor,
    encode: Transform,
    decode: Transform,
    latent_distribution: Distribution,
    manifold: Manifold | None = None,
) -> NLLExactOutput:
    """Computes the exact negative log likelihood, by computing the
    exact volume change term in the change of variables formula under
    the decode function.

    :param x: Input data. Shape: (batch_size, ...)
    :param encode: Encoder function. Takes `x` as input and returns a latent
        representation `z` of shape (batch_size, latent_shape).
    :param decode: Decoder function. Takes a latent representation `z` as input
        and returns a reconstruction `x1`.
    :param latent_distribution: The latent distribution to use.
    :param manifold: Optional manifold the encoder and decoder should be
        restricted to. If provided, the volume change term is computed
        in the tangent space of this manifold.
    :return: The negative log likelihood if shape (batch_size,), the latent
        representation `z`, the reconstruction `x1` and additional
        regularizations computed on the fly.
    """
    exact = volume_change_exact(x, encode, decode, manifold)
    nll = -latent_distribution.log_prob(exact.z)
    nll -= exact.exact
    if manifold is not None:
        nll -= volume_change_metric(x, exact.z, manifold)

    return NLLExactOutput(nll, exact.z, exact.x1, exact.regularizations)
