from .earth import get_earth_dataset
from .image import (get_celeba_datasets, get_cifar10_datasets,
                    get_mnist_datasets)
from .molecular import (load_dw4_dataset, load_lj13_dataset, load_lj55_dataset,
                        load_qm9_dataset, make_2d_atom_grid_datasets)
from .sbi import get_sbi_dataset
from .tabular import get_tabular_datasets
from .torus import get_torus_protein_dataset, get_torus_rna_dataset
from .toy import make_toy_data
from .utils import TrainValTest

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
    elif name == "special-orthogonal":
        from fff.data.special_orthogonal import make_so_data
        return make_so_data(**kwargs)
    elif name == "torus_protein":
        return get_torus_protein_dataset(**kwargs)
    elif name == "torus_rna":
        return get_torus_rna_dataset(**kwargs)
    elif name in ["fire", "flood", "quakes", "volcano"]:
        return get_earth_dataset(name, **kwargs)
    else:
        return make_toy_data(name, **kwargs)
