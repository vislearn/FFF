from typing import Callable
import torch

Transform = Callable[[torch.Tensor], torch.Tensor]
