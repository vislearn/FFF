from functools import partial, wraps
import torch
from torch.func import jacfwd, jacrev, vmap

from fff.model.utils import batch_wrap


def double_output(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        return out, out

    return wrapper


def compute_jacobian(x_in, fn, *func_args, chunk_size=None, grad_type="backward", **func_kwargs):
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


@torch.no_grad()
def compute_volume_change(func, input):
    output, jac = compute_jacobian(input, func)

    if jac.shape[-1] < jac.shape[-2]:
        jac_prod_fn = lambda x: x.T @ x
    else:
        jac_prod_fn = lambda x: x @ x.T

    jac_prod = vmap(jac_prod_fn)(jac)
    return output, torch.slogdet(jac_prod)[1]
