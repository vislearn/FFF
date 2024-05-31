from typing import Callable, Any, Tuple

import torch
from torch.func import jacfwd, jacrev, vmap

from functools import partial, wraps

from fff.utils.utils import batch_wrap


def double_output(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        return out, out

    return wrapper


def compute_jacobian(
    x_in: torch.Tensor,
    fn: Callable,
    *func_args: Any,
    chunk_size: int = None,
    grad_type: str = "backward",
    **func_kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the jacobian of a function with respect to its input.

    :param x_in: The input tensor to compute the jacobian at.
        Shape: (batch_size, in_dim).
    :param fn: The function to compute the jacobian of, which transforms
        `x` to `fn(x)` of shape (batch_size, out_dim).
    :param func_args: The positional arguments to pass to the function.
        func_args are batched over the first dimension.
    :param chunk_size: The chunk size to use for batching.
    :param grad_type: The type of gradient to use. Either 'backward' or
        'forward'.
    :param func_kwargs: The keyword arguments to pass to the function.
        func_kwargs are not batched.
    :return: The output of the function `fn(x)` and the jacobian of the
        function with respect to its input `x` of shape
        (batch_size, out_dim, in_dim)."""
    jacfn = jacrev if grad_type == "backward" else jacfwd
    with torch.inference_mode(False):
        with torch.no_grad():
            fn_kwargs_prefilled = partial(fn, **func_kwargs)
            fn_batch_expanded = batch_wrap(fn_kwargs_prefilled)
            fn_return_val = double_output(fn_batch_expanded)
            fn_jac_batched = vmap(
                jacfn(fn_return_val, has_aux=True), chunk_size=chunk_size
            )
            jac, x_out = fn_jac_batched(x_in, *func_args)
    return x_out, jac


def compute_volume_change(jac: torch.Tensor) -> torch.Tensor:
    """Computes the volume change of a function given its jacobian.

    :param jac: The jacobian of the function. of shape
        (batch_size, out_dim, in_dim).
    :return: The volume change of the function. of shape (batch_size,)."""
    full_dimensional = jac.shape[-1] == jac.shape[-2]
    if full_dimensional:
        return jac.slogdet()[1]
    else:
        if jac.shape[-1] < jac.shape[-2]:
            jac = jac.transpose(-1, -2)
        jac_transpose_jac = torch.bmm(jac, jac.transpose(1, 2))
        return jac_transpose_jac.slogdet()[1] / 2
