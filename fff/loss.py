# MIT License
#
# Copyright (c) 2024 Computer Vision and Learning Lab, Heidelberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import functools
from collections import namedtuple
from functools import wraps, partial
from math import sqrt, prod
from typing import Union
from geomstats.geometry.manifold import Manifold

import torch
from torch.distributions import Distribution
from torch.autograd import grad
from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual

from fff.utils.geometry import random_tangent_vec
from fff.utils.utils import sum_except_batch, fix_device
from fff.utils.types import Transform

SurrogateOutput = namedtuple(
    "SurrogateOutput", ["surrogate", "z", "x1", "regularizations"]
)


def sample_v(
    x: torch.Tensor, hutchinson_samples: int, manifold: Manifold | None = None
) -> torch.Tensor:
    """
    Sample a random vector v of shape (*x.shape, hutchinson_samples)
    with scaled orthonormal columns.

    The reference data is used for shape, device and dtype.

    :param x: Reference data. Shape: (batch_size, ...).
    :param hutchinson_samples: Number of Hutchinson samples to draw.
    :param manifold: Optional manifold on which the data lies. If provided,
        the vectors are sampled in the tangent space of the manifold.
    :return: Random vectors of shape (batch_size, ...)
    """
    batch_size, total_dim = x.shape[0], prod(x.shape[1:])

    if hutchinson_samples > total_dim:
        raise ValueError(
            f"Too many Hutchinson samples: got {hutchinson_samples}, \
                expected <= {total_dim}"
        )
    # M-FFF: More than one Hutchinson sample not implemented for M-FFF
    if manifold is not None and hutchinson_samples != 1:
        raise NotImplementedError(
            f"More than one Hutchinson sample not implemented for M-FFF, \
                {hutchinson_samples} requested."
        )

    if manifold is None:
        v = torch.randn(
            batch_size, total_dim, hutchinson_samples, device=x.device, dtype=x.dtype
        )
        q = torch.linalg.qr(v).Q.reshape(*x.shape, hutchinson_samples)
        return q * sqrt(total_dim)
    # M-FFF: Sample v in the tangent space of the manifold at x
    else:
        v = random_tangent_vec(manifold, x.detach(), n_samples=batch_size)
        v /= torch.norm(v, p=2, dim=list(range(1, len(v.shape))), keepdim=True)
        return v[..., None] * sqrt(total_dim)


def volume_change_surrogate(
    x: torch.Tensor,
    encode: Transform,
    decode: Transform,
    hutchinson_samples: int = 1,
    manifold: Manifold | None = None,
) -> SurrogateOutput:
    r"""Computes the surrogate for the volume change term in the change of
    variables formula. The surrogate is given by:
    $$
    v^T f_\theta'(x) \texttt{SG}(g_\phi'(z) v).
    $$
    The gradient of the surrogate is the gradient of the volume change term.

    :param x: Input data. Shape: (batch_size, ...)
    :param encode: Encoder function. Takes `x` as input and returns a latent
        representation `z` of shape (batch_size, latent_shape).
    :param decode: Decoder function. Takes a latent representation `z` as input
        and returns a reconstruction `x1`.
    :param hutchinson_samples: Number of Hutchinson samples to use for the
        volume change estimator. The number of hutchinson samples must be less
        than or equal to the total dimension of the data.
    :param manifold: Manifold on which the latent space lies. If provided, the
        volume change is computed in the tangent space of the manifold.
    :return: The computed surrogate of shape (batch_size,), latent representation
        `z`, reconstruction `x1` and regularization metrics computed on the fly.
    """
    regularizations = {}
    surrogate = 0

    x.requires_grad_()
    z = encode(x)

    # M-FFF: Project to manifold and store projection distance for
    # regularization
    if manifold is not None:
        z_projected = manifold.projection(z)
        regularizations["z_projection"] = reconstruction_loss(z, z_projected)
        z = z_projected

    vs = sample_v(z, hutchinson_samples, manifold)

    for k in range(hutchinson_samples):
        v = vs[..., k]

        # $ g'(z) v $ via forward-mode AD
        with dual_level():
            dual_z = make_dual(z, v)
            dual_x1 = decode(dual_z)

            # M-FFF: Project to manifold and store projection distance for
            # regularization
            if manifold is not None:
                dual_x1_projected = manifold.projection(dual_x1)
                regularizations["x1_projection"] = reconstruction_loss(
                    unpack_dual(dual_x1_projected)[0], unpack_dual(dual_x1)[0]
                )
                dual_x1 = dual_x1_projected

            x1, v1 = unpack_dual(dual_x1)

        # $ v^T f'(x) $ via backward-mode AD
        (v2,) = grad(z, x, v, create_graph=True)

        # $ v^T f'(x) stop_grad(g'(z)) v $
        surrogate += sum_except_batch(v2 * v1.detach()) / hutchinson_samples

    return SurrogateOutput(surrogate, z, x1, regularizations)


def volume_change_metric(
    x: torch.Tensor, z: torch.Tensor, manifold: Manifold
) -> torch.Tensor:
    r"""Compute the volume change term induced by the metric,
    which is given by:
    $$
    \tfrac{1}{2} \log \frac{|R^T G(f(x)) R|}{|Q^T G(x) Q|},
    $$
    where $G(f(x))$ and $G(x)$ are the metric matrices at the latent
    representation $f(x) = z$ and the data $x$, respectively.

    :param x: Input data. Shape: (batch_size, ...)
    :param z: Latent representation. Shape: (batch_size, latent_shape)
    :param manifold: Manifold. The `default_metric` method of the manifold
        should return the metric that should be used for the volume change
        computation. The metric must have a `metric_matrix_log_det` method
        implemented, which returns the log determinant of the metric matrix.
    :return: Volume change term. Shape: (batch_size,)
    """

    metric = manifold.default_metric()(manifold)
    if hasattr(metric, "metric_matrix_log_det"):
        metric_volume_change = 0.5 * (
            metric.metric_matrix_log_det(z) - metric.metric_matrix_log_det(x)
        )
        return sum_except_batch(metric_volume_change)

    return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)


def reconstruction_loss(
    a: torch.Tensor, b: torch.Tensor, manifold: Manifold | None = None
) -> torch.Tensor:

    if manifold is not None:
        metric = manifold.default_metric()(manifold)
        return fix_device(metric.squared_dist)(a, b)
    else:
        return torch.sum((a - b) ** 2, dim=tuple(range(1, len(a.shape))))


def fff_loss(
    x: torch.Tensor,
    encode: Transform,
    decode: Transform,
    latent_distribution: Distribution,
    beta: Union[float, torch.Tensor],
    hutchinson_samples: int = 1,
) -> torch.Tensor:
    r"""Compute the per-sample FFF/FIF loss:
    $$
    \mathcal{L} = \beta ||x - decode(encode(x))||^2 - \log p_Z(z)
        - \sum_{k=1}^K v_k^T f'(x) stop_grad(g'(z)) v_k
    $$
    where $E[v_k^T v_k] = 1$, and $ f'(x) $ and $ g'(z) $ are the Jacobians of
    `encode` and `decode`.

    :param x: Input data. Shape: (batch_size, ...)
    :param encode: Encoder function. Takes `x` as input and returns a latent
        representation `z` of shape (batch_size, latent_shape).
    :param decode: Decoder function. Takes a latent representation `z` as input
        and returns a reconstruction `x1`.
    :param latent_distribution: Latent distribution of the model.
    :param beta: Weight of the mean squared error.
    :param hutchinson_samples: Number of Hutchinson samples to use for the
        volume change estimator.
    :return: Per-sample loss. Shape: (batch_size,)"""
    surrogate = volume_change_surrogate(x, encode, decode, hutchinson_samples)
    mse = reconstruction_loss(x, surrogate.x1)
    log_prob = latent_distribution.log_prob(surrogate.z)
    nll = -sum_except_batch(log_prob) - surrogate.surrogate
    return beta * mse + nll


def mfff_loss(
    x: torch.Tensor,
    encode: Transform,
    decode: Transform,
    latent_distribution: Distribution,
    beta: Union[float, torch.Tensor],
    manifold: Manifold,
    hutchinson_samples: int = 1,
    manifold_distance: bool = False,
) -> torch.Tensor():
    r"""Compute the per sample M-FFF loss:
    $$
    \mathcal{L} = \beta ||x - decode(encode(x))||^2 - \log p_Z(z)
        - v^T f_\theta'(x) \texttt{SG}[{f_\theta^{-1}}'(z)v
        - \tfrac{1}{2} \log \frac{|R^T G(f_\theta(x)) R|}{|Q^T G(x) Q|}
    $$
    where $E[v_k^T v_k] = 1$, and $ f'(x) $ and $ g'(z) $ are the Jacobians of
    `encode` and `decode`.

    :param x: Input data. Shape: (batch_size, ...)
    :param encode: Encoder function. Takes `x` as input and returns a latent
        representation `z` of shape (batch_size, latent_shape).
    :param decode: Decoder function. Takes a latent representation `z` as input
        and returns a reconstruction `x1`.
    :param latent_distribution: Latent distribution of the model.
    :param beta: Weight of the mean squared error.
    :param manifold: Manifold on which encode and decode act.
    :param hutchinson_samples: Number of Hutchinson samples to use for the
        volume change estimator.
    :param manifold_distance: If True, use the manifold distance for the
        reconstruction loss. Otherwise, use the Euclidean distance.
    :return: Per-sample loss. Shape: (batch_size,)"""
    surrogate = volume_change_surrogate(x, encode, decode, hutchinson_samples, manifold)
    if manifold_distance:
        mse = reconstruction_loss(x, surrogate.x1, manifold)
    else:
        mse = reconstruction_loss(x, surrogate.x1)
    log_prob = latent_distribution.log_prob(surrogate.z)
    metric_vol_change = volume_change_metric(x, surrogate.z, manifold)
    nll = -sum_except_batch(log_prob) - surrogate.surrogate - metric_vol_change
    return beta * mse + nll
