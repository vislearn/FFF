from .image import get_mnist_datasets, get_cifar10_datasets, get_celeba_datasets
from .toy import make_toy_data
from .utils import TrainValTest
from .tabular import get_tabular_datasets
from .molecular import make_2d_atom_grid_datasets, load_qm9_dataset, load_dw4_dataset, load_lj13_dataset, \
    load_lj55_dataset
from .sbi import get_sbi_dataset

__all__ = ["load_dataset"]


def load_dataset(name: str, **kwargs) -> TrainValTest:
    if name in ["miniboone", "gas", "hepmass", "power"]:
        # note that the given train/val/test split is ignored and a fixed split is performed
        return get_tabular_datasets(name=name, **kwargs)
    elif name == "mnist":
        return get_mnist_datasets(**kwargs)
    elif name == "cifar10":
        return get_cifar10_datasets(**kwargs)
    elif name == "celeba":
        return get_celeba_datasets(**kwargs)
    elif name == "mol-grid":
        return make_2d_atom_grid_datasets(**kwargs)
    elif name == "qm9":
        return load_qm9_dataset(**kwargs)
    elif name == "dw4":
        return load_dw4_dataset(**kwargs)
    elif name == "lj13":
        return load_lj13_dataset(**kwargs)
    elif name == "lj55":
        return load_lj55_dataset(**kwargs)
    elif name.startswith("sbi_"):
        parts = name.split("_")
        taskname = "_".join(parts[1:])
        return get_sbi_dataset(name=taskname, **kwargs)
    else:
        return make_toy_data(name, **kwargs)
