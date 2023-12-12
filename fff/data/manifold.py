try:
    from geomstats.geometry.manifold import Manifold
except ImportError:
    Manifold = object
from torch.utils.data import Dataset
import torch
from functools import wraps


class ManifoldDataset(Dataset):
    def __init__(self, dataset: Dataset, manifold: Manifold):
        self.dataset = dataset
        self.manifold = manifold

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


def fix_device(fun):
    @wraps(fun)
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
