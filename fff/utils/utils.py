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
    :param x: Input tensor. Shape: (batch_size, ...)
    :return: Sum over all dimensions except the first. Shape: (batch_size,)
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


def fix_device(fun):
    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        # Guess device, dtype from arguments
        devices = set(arg.device for arg in args if isinstance(arg, torch.Tensor))
        assert len(devices) == 1, "Multiple devices in arguments"
        dtypes = set(arg.dtype for arg in args if isinstance(arg, torch.Tensor))
        assert len(dtypes) == 1, "Multiple devices in arguments"
        device, = devices
        dtype, = dtypes

        try:
            default_device_type = torch.Tensor().device.type
            default_dtype = torch.Tensor().dtype
            if default_device_type != device.type:
                if device.type == "cuda":
                    torch.set_default_tensor_type(
                        torch.cuda.FloatTensor
                        if dtype == torch.float32
                        else torch.cuda.DoubleTensor
                    )
                if device.type == "cpu":
                    torch.set_default_tensor_type(
                        torch.FloatTensor
                        if dtype == torch.float32
                        else torch.DoubleTensor
                    )
            return fun(*args, **kwargs)
        finally:
            if default_device_type != device.type:
                if default_device_type == "cuda":
                    torch.set_default_tensor_type(
                        torch.cuda.FloatTensor
                        if default_dtype == torch.float32
                        else torch.cuda.DoubleTensor
                    )
                if default_device_type == "cpu":
                    torch.set_default_tensor_type(
                        torch.FloatTensor
                        if default_dtype == torch.float32
                        else torch.DoubleTensor
                    )

    return wrapper
