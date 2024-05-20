import os
from typing import Tuple, Type

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
