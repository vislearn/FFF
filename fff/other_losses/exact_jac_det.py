from collections import namedtuple
from functools import wraps, partial
from math import prod

import torch

from fff.model.utils import batch_wrap

try:
    from torch.func import vmap, jacrev, jacfwd
except ImportError:
    from functorch import vmap, jacrev, jacfwd




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
