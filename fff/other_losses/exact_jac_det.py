from collections import namedtuple
from functools import wraps, partial
from math import prod

import torch

from fff.model.utils import batch_wrap

try:
    from torch.func import vmap, jacrev, jacfwd
except ImportError:
    from functorch import vmap, jacrev, jacfwd


def double_output(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        return out, out

    return wrapper


ExactOutput = namedtuple("ExactOutput", ["z", "x1", "nll", "log_det", "regularizations"])


def compute_volume_change(jac):
    full_dimensional = jac.shape[-1] == jac.shape[-2]
    if full_dimensional:
        return jac.slogdet()[1]
    else:
        if jac.shape[-1] < jac.shape[-2]:
            jac = jac.transpose(-1, -2)
        jac_transpose_jac = torch.bmm(jac, jac.transpose(1, 2))
        return jac_transpose_jac.slogdet()[1] / 2



def log_det_exact(x, encode, decode, *func_args,
                  grad_type="backward", jacobian_target="encoder",
                  chunk_size=None, allow_gradient=False, **func_kwargs) -> ExactOutput:
    """
    Compute the exact log determinant of the Jacobian of the given encoder or decoder.

    :param x: The input batch.
    :param encode: The encoder function, is called as `encode(x, *func_args, **func_kwargs)`
    :param decode: The decoder function, is called as `decode(z, *func_args, **func_kwargs)`
    :param grad_type: Should the Jacobian be computed using forward or backward mode AD?
    :param jacobian_target: Should the Jacobian of the encoder or decoder be computed?
    :param chunk_size: Set to a positive integer to enable chunking of the computation.
    :param allow_gradient: Should gradients be allowed to flow through the computation?
    :param func_args: Additional arguments to encode/decode
    :param func_kwargs: Additional keyword arguments to encode/decode
    :return:
    """
    if torch.is_grad_enabled() and not allow_gradient:
        raise RuntimeError("Exact log det computation is only recommended in torch.no_grad() mode "
                           "as training may be unstable (see Section 4.2 in the FIF paper). "
                           "Set log_det_estimator.allow_gradient=True to allow computing gradients.")
    jacfn = jacrev if grad_type == "backward" else jacfwd

    batch_size = x.shape[0]
    n_in_dim = prod(x.shape[1:])

    if jacobian_target not in ["encoder", "decoder", "both"]:
        raise ValueError(f"Unknown jacobian target: {jacobian_target}")

    # Encoder jacobian if requested
    if jacobian_target in ["encoder", "both"]:
        with torch.inference_mode(False):
            with torch.no_grad():
                jac_enc, z = vmap(jacfn(double_output(batch_wrap(partial(encode, **func_kwargs))), has_aux=True),
                                  chunk_size=chunk_size)(x, *func_args)
        n_out_dim = prod(z.shape[1:])
        jac_enc = jac_enc.reshape(batch_size, n_out_dim, n_in_dim)
    else:
        z = encode(x, *func_args, **func_kwargs)
        n_out_dim = prod(z.shape[1:])
        jac_enc = None

    # Decoder jacobian if requested
    if jacobian_target in ["decoder", "both"]:
        with torch.inference_mode(False):
            with torch.no_grad():
                jac_dec, x1 = vmap(jacfn(double_output(batch_wrap(partial(decode, **func_kwargs))), has_aux=True),
                                   chunk_size=chunk_size)(z, *func_args)
        jac_dec = jac_dec.reshape(batch_size, n_out_dim, n_in_dim)
    else:
        x1 = decode(z, *func_args, **func_kwargs)
        jac_dec = None

    # Encoder is default if both are computed
    metrics = {}
    if jac_dec is not None:
        vol_change = metrics["vol_change_dec"] = -compute_volume_change(jac_dec)
    if jac_enc is not None:
        vol_change = metrics["vol_change_enc"] = compute_volume_change(jac_enc)

    # Default NLL with unnormalized normal distribution
    nll = torch.sum((z.reshape(batch_size, n_out_dim) ** 2), -1) / 2 - vol_change

    return ExactOutput(
        z, x1, nll, vol_change, metrics
    )
