# MIT License
#
# Copyright (c) 2023 Computer Vision and Learning Lab, Heidelberg
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

from collections import namedtuple
from math import sqrt, prod
from typing import Union, Callable

import torch
from FrEIA.utils import sum_except_batch
from torch.autograd import grad
from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual

SurrogateOutput = namedtuple("SurrogateOutput", ["z", "x1", "nll", "surrogate", "regularizations"])
Transform = Callable[[torch.Tensor], torch.Tensor]


def sample_v(x: torch.Tensor, hutchinson_samples: int):
    """
    Sample a random vector v of shape (*x.shape, hutchinson_samples)
    with scaled orthonormal columns.

    The reference data is used for shape, device and dtype.

    :param x: Reference data.
    :param hutchinson_samples: Number of Hutchinson samples to draw.
    :return:
    """
    batch_size, total_dim = x.shape[0], prod(x.shape[1:])
    if hutchinson_samples > total_dim:
        raise ValueError(f"Too many Hutchinson samples: got {hutchinson_samples}, expected <= {total_dim}")
    v = torch.randn(batch_size, total_dim, hutchinson_samples, device=x.device, dtype=x.dtype)
    q = torch.linalg.qr(v).Q.reshape(*x.shape, hutchinson_samples)
    return q * sqrt(total_dim)


def mat_mul_fns(x, forward):
    def at_vec_fn(v):
        """
        Compute the Jacobian-vector product Jv using torch.autograd.grad.
        """
        x.requires_grad_(True)
        y = forward(x)
        jvp = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=v, retain_graph=True, create_graph=True)[0]
        return jvp.detach()

    def a_vec_fn(v):
        """
        Compute the vector-Jacobian product v^T J.
        """
        x.requires_grad_(True)
        # Using forward mode autodiff
        with dual_level():
            dual_x = make_dual(x, v)
            dual_y = forward(dual_x)
            y, vjp = unpack_dual(dual_y)

        return vjp.detach()

    return a_vec_fn, at_vec_fn


def cg_normal_eq(b, x0, a_vec_fn, at_vec_fn, lambda_reg=0., max_iter=1000, tol=1e-6):
    """
    Conjugate Gradient (CG) method to solve the normal equations using given functions.

    Parameters:
    - b: Right-hand side tensor of shape (batch_size, m)
    - a_vec_fn: Function to compute the product Ax for x of shape (batch_size, n)
    - at_vec_fn: Function to compute the product A^T r for r of shape (batch_size, m)
    - max_iter: Maximum number of iterations
    - tol: Tolerance for convergence

    Returns:
    - x: Solution tensor of shape (batch_size, n)
    - residuals: List of residuals for each iteration
    """
    # Compute ATb only once
    ATb = at_vec_fn(b)
    x = x0
    r = ATb - (at_vec_fn(a_vec_fn(x)) + lambda_reg * x)
    p = r

    for _ in range(max_iter):
        r_norm = torch.norm(r, dim=1)

        if torch.max(r_norm) < tol:
            break

        Ap = at_vec_fn(a_vec_fn(p)) + lambda_reg * p

        alpha = torch.sum(r * r, dim=1) / torch.sum(p * Ap, dim=1)
        x = x + alpha.unsqueeze(1) * p
        r_new = r - alpha.unsqueeze(1) * Ap

        beta = torch.sum(r_new * r_new, dim=1) / torch.sum(r * r, dim=1)
        p = r_new + beta.unsqueeze(1) * p

        r = r_new

    return x


def nll_surrogate(x: torch.Tensor, encode: Transform, decode: Transform,
                  hutchinson_samples: int = 1,
                  cg_n_iter: int = 0, cg_reg: float = 0.0,
                  compute_landweber_step: bool = False) -> SurrogateOutput:
    """
    Compute the per-sample surrogate for the negative log-likelihood and the volume change estimator.
    The gradient of the surrogate is the gradient of the actual negative log-likelihood.

    :param x: Input data. Shape: (batch_size, ...)
    :param encode: Encoder function. Takes `x` as input and returns a latent representation of shape (batch_size, latent_dim).
    :param decode: Decoder function. Takes a latent representation of shape (batch_size, latent_dim) as input and returns a reconstruction of shape (batch_size, ...).
    :param hutchinson_samples: Number of Hutchinson samples to use for the volume change estimator.
    :return: Per-sample loss. Shape: (batch_size,)
    """
    x.requires_grad_()
    z = encode(x)

    regularizations = {}
    v1s = []
    v2s = []
    if cg_n_iter > 0:
        regularizations["cg_g_guidance"] = 0

    surrogate = 0
    vs = sample_v(z, hutchinson_samples)
    for k in range(hutchinson_samples):
        v = vs[..., k]

        # $ g'(z) v $ via forward-mode AD
        with dual_level():
            dual_z = make_dual(z, v)
            dual_x1 = decode(dual_z)
            x1, v1 = unpack_dual(dual_x1)

        # Improve the estimate of $ f'(z)^+ v $ using CG
        if cg_n_iter > 0:
            a_vec_fn, at_vec_fn = mat_mul_fns(x, encode)
            v1_0 = v1
            v1 = cg_normal_eq(v, v1, a_vec_fn, at_vec_fn,
                              lambda_reg=cg_reg, max_iter=cg_n_iter)

            # Guide decoder closer to CG solution
            regularizations["cg_g_guidance"] += sum_except_batch((v1.detach() - v1_0) ** 2)

        # Do one Landweber step via decoder gradient descent
        if compute_landweber_step:
            a_vec_fn, at_vec_fn = mat_mul_fns(x, encode)
            regularizations["landweber"] = sum_except_batch((v - a_vec_fn(v1)) ** 2)

        v1s.append(v1)

        # $ v^T f'(x) $ via backward-mode AD
        v2, = grad(z, x, v, create_graph=True)
        v2s.append(v2)

        # $ v^T f'(x) stop_grad(g'(z)) v $
        surrogate += sum_except_batch(v2 * v1.detach()) / hutchinson_samples

    # In basis of v, f' g' should be the identity times d
    regularizations["fg_orthogonality"] = 0.0
    if hutchinson_samples > 1:
        # f' g' should be symmetric
        regularizations["fg_symmetry"] = 0.0
    for i in range(len(v1s)):
        for j in range(len(v2s)):
            if i != j:
                regularizations[f"fg_orthogonality"] += sum_except_batch(v1s[i] * v2s[j]) ** 2
                if "fg_symmetry" in regularizations:
                    dot_a = sum_except_batch(v1s[i] * v2s[j])
                    dot_b = sum_except_batch(v2s[i] * v1s[j])
                    regularizations["fg_symmetry"] += (dot_a - dot_b) ** 2
            else:
                regularizations[f"fg_orthogonality"] += (sum_except_batch(v1s[i] * v2s[i]) - z.shape[-1]) ** 2

    # Per-sample negative log-likelihood
    nll = sum_except_batch((z ** 2)) / 2 - surrogate

    return SurrogateOutput(z, x1, nll, surrogate, regularizations)


def fff_loss(x: torch.Tensor,
              encode: Transform, decode: Transform,
              beta: Union[float, torch.Tensor],
              hutchinson_samples: int = 1) -> torch.Tensor:
    """
    Compute the per-sample FFF/FIF loss:
    $$
    \mathcal{L} = \beta ||x - decode(encode(x))||^2 + ||encode(x)||^2 // 2 - \sum_{k=1}^K v_k^T f'(x) stop_grad(g'(z)) v_k
    $$
    where $E[v_k^T v_k] = 1$, and $ f'(x) $ and $ g'(z) $ are the Jacobians of `encode` and `decode`.

    :param x: Input data. Shape: (batch_size, ...)
    :param encode: Encoder function. Takes `x` as input and returns a latent representation of shape (batch_size, latent_dim).
    :param decode: Decoder function. Takes a latent representation of shape (batch_size, latent_dim) as input and returns a reconstruction of shape (batch_size, ...).
    :param beta: Weight of the mean squared error.
    :param hutchinson_samples: Number of Hutchinson samples to use for the volume change estimator.
    :return: Per-sample loss. Shape: (batch_size,)
    """
    surrogate = nll_surrogate(x, encode, decode, hutchinson_samples)
    mse = torch.sum((x - surrogate.x1) ** 2, dim=tuple(range(1, len(x.shape))))
    return beta * mse + surrogate.nll
