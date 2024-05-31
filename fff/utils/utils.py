import os
from typing import Tuple, Type, Callable

import functools

import torch
from torch import nn


def get_latest_run(
    model_type: Type[nn.Module], checkpoint_path: str, **kwargs
) -> Tuple[nn.Module, int]:
    """Returns the latest run from the checkpoint path
    :param checkpoint_path: the path to the checkpoint
    :return: the model and the step of the latest run"""

    latest = 0
    latest_name = None
    for file in os.listdir(checkpoint_path):
        if file.endswith(".ckpt"):
            ckpt_name = file.split(".")[0]
            try:
                ckpt_step = int(ckpt_name.split("step=")[-1])
            except ValueError:
                continue
            if ckpt_step > latest:
                latest = ckpt_step
                latest_name = ckpt_name
    model = model_type.load_from_checkpoint(
        os.path.join(checkpoint_path, latest_name + ".ckpt"), **kwargs
    )
    return model, latest


def sum_except_batch(x: torch.Tensor) -> torch.Tensor:
    """Sum over all dimensions except the first.
    :param x: Input tensor.
    :return: Sum over all dimensions except the first.
    """
    return torch.sum(x.reshape(x.shape[0], -1), dim=1)


def batch_wrap(fn: Callable) -> Callable:
    """Add a batch dimension to each tensor argument.

    :param fn:
    :return:"""

    def deep_unsqueeze(arg):
        if torch.is_tensor(arg):
            return arg[None, ...]
        elif isinstance(arg, dict):
            return {key: deep_unsqueeze(value) for key, value in arg.items()}
        elif isinstance(arg, (list, tuple)):
            return [deep_unsqueeze(value) for value in arg]

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        args = deep_unsqueeze(args)
        return fn(*args, **kwargs)[0]

    return wrapper
