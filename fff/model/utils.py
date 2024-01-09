import functools
import time
from inspect import isclass
from math import sqrt

import FrEIA
import torch
from lightning import Callback
from torch import nn


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)
    
class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)
    
torch.nn.Sin = Sin
torch.nn.Swish = Swish

def get_module(name):
    """ Get a nn.Module in a case-insensitive way """
    modules = torch.nn.__dict__
    modules = {
        key.lower(): value for key, value in modules.items()
        if isclass(value) and issubclass(value, torch.nn.Module)
    }

    return modules[name.lower()]


def make_dense(widths: list[int], activation: str, dropout: float = None, batch_norm: str | bool = False):
    """ Make a Dense Network from given layer widths and activation function """
    if len(widths) < 2:
        raise ValueError("Need at least Input and Output Layer.")

    Activation = get_module(activation)

    network = nn.Sequential()

    # input is x, time, condition
    for i in range(0, len(widths) - 2):
        if i > 0 and dropout is not None:
            network.add_module(f"Dropout_{i}", nn.Dropout1d(p=dropout))
        hidden_layer = nn.Linear(in_features=widths[i], out_features=widths[i + 1])
        network.add_module(f"Hidden_Layer_{i}", hidden_layer)
        if batch_norm is not False:
            network.add_module(f"Batch_Norm_{i}", wrap_batch_norm1d(batch_norm, widths[i + 1]))
        network.add_module(f"Hidden_Activation_{i}", Activation())

    # output is velocity
    output_layer = nn.Linear(in_features=widths[-2], out_features=widths[-1])
    network.add_module("Output_Layer", output_layer)

    return network


def guess_image_shape(dim):
    if dim == 3 * 38804:
        return 3, 178, 218
    if dim % 3 == 0:
        n_channels = 3
    else:
        n_channels = 1
    size = round(sqrt(dim // n_channels))
    if size ** 2 * n_channels != dim:
        raise ValueError(f"Input is not square: "
                         f"{size} ** 2 != {dim // n_channels}")
    return n_channels, size, size


def subnet_factory(inner_widths, activation, zero_init=True):
    def make_subnet(dim_in, dim_out):
        network = make_dense([
            dim_in,
            *inner_widths,
            dim_out
        ], activation)

        if zero_init:
            network[-1].weight.data.zero_()
            network[-1].bias.data.zero_()

        return network

    return make_subnet


def make_inn(inn_spec, *data_dim, zero_init=True):
    inn = FrEIA.framework.SequenceINN(*data_dim)
    for inn_layer in inn_spec:
        module_name, module_args, subnet_widths = inn_layer
        module_class = getattr(FrEIA.modules, module_name)
        extra_module_args = dict()
        if "subnet_constructor" not in module_args:
            extra_module_args["subnet_constructor"] = subnet_factory(subnet_widths, "silu", zero_init=zero_init)
        inn.append(module_class, **module_args, **extra_module_args)
    return inn


def batch_wrap(fn):
    """
    Add a batch dimension to each tensor argument.

    :param fn:
    :return:
    """

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


class RunningBatchNorm(torch.nn.Module):
    """
    Wrap BatchNorm to normalize only using running mean and std,
    instead of per-batch normalization.
    """

    def __init__(self, train_batch_norm, eval_batch_norm):
        super().__init__()
        self.train_batch_norm = train_batch_norm
        self.eval_batch_norm = eval_batch_norm

    def forward(self, x):
        # Apply BatchNorms, updating their running mean/std
        if self.training:
            try:
                self.train_batch_norm(x)
                if self.train_batch_norm is not self.eval_batch_norm:
                    self.eval_batch_norm(x)
            except ValueError:
                pass

        # Use result from aggregated mean and std for actual computation
        if self.training:
            self.train_batch_norm.eval()
            out = self.train_batch_norm(x)
            self.train_batch_norm.train()
        else:
            try:
                out = self.eval_batch_norm(x)
            except ValueError:
                # In vmap calls, fake a batch dimension
                out = self.eval_batch_norm(x[None])[0]
        return out


class VmapBatchNorm(torch.nn.Module):
    def __init__(self, batch_norm):
        super().__init__()
        self.batch_norm = batch_norm

    def forward(self, x):
        try:
            return self.batch_norm(x)
        except ValueError:
            if self.training:
                raise
            return self.batch_norm(x[None])[0]


def _make_batch_norm(kind):
    @functools.wraps(kind)
    def wrapper(batch_norm_spec: str | bool, *args, **kwargs):
        assert batch_norm_spec is not False
        batch_norm = kind(*args, **kwargs)
        if batch_norm_spec == "no-batch-grads":
            # This mode behaves like traditional BatchNorm, but it ignores
            # gradients between batch entries
            train_batch_norm = kind(*args, momentum=1.0, **kwargs)
            batch_norm = RunningBatchNorm(train_batch_norm, batch_norm)
        elif batch_norm_spec == "running-only":
            # This mode keeps track of running stats
            batch_norm = RunningBatchNorm(batch_norm, batch_norm)
        elif batch_norm_spec == "vmap":
            # This mode uses unpacks vmap tensors
            batch_norm = VmapBatchNorm(batch_norm)
        elif batch_norm_spec is not True:
            raise ValueError(f"{batch_norm_spec=}")
        return batch_norm

    return wrapper


wrap_batch_norm1d = _make_batch_norm(nn.BatchNorm1d)
wrap_batch_norm2d = _make_batch_norm(nn.BatchNorm2d)
wrap_batch_norm3d = _make_batch_norm(nn.BatchNorm3d)


class TrainWallClock(Callback):
    def __init__(self):
        self.batch_start = None
        self.state = {"steps": 0, "time": 0}

    def on_train_batch_start(self, *args, **kwargs) -> None:
        self.batch_start = time.monotonic()

    def on_train_batch_end(self, *args, **kwargs):
        self.state["steps"] += 1
        self.state["time"] += time.monotonic() - self.batch_start
        self.batch_start = None

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

    def state_dict(self):
        return self.state.copy()
