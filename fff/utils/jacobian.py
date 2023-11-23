from functools import partial
import torch
from torch.func import jacfwd, jacrev, vmap

from fff.model.utils import batch_wrap
from fff.other_losses.exact_jac_det import double_output


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
