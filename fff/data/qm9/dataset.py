from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset
from fff.data.qm9.data.args import init_argparse
from fff.data.qm9.data.collate import PreprocessQM9
from fff.data.qm9.data.utils import initialize_datasets
import os


def retrieve_dataloaders(cfg):
    if 'qm9' in cfg.dataset:
        batch_size = cfg.batch_size
        num_workers = cfg.num_workers
        filter_n_atoms = cfg.filter_n_atoms
        # Initialize dataloader
        args = init_argparse('qm9')
        # data_dir = cfg.data_root_dir
        args, datasets, num_species, charge_scale = initialize_datasets(args, cfg.datadir, cfg.dataset,
                                                                        subtract_thermo=args.subtract_thermo,
                                                                        force_download=args.force_download,
                                                                        remove_h=cfg.remove_h)
        qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114,
                     'homo': 27.2114,
                     'lumo': 27.2114}

        for dataset in datasets.values():
            dataset.convert_units(qm9_to_eV)

        if filter_n_atoms is not None:
            print("Retrieving molecules with only %d atoms" % filter_n_atoms)
            datasets = filter_atoms(datasets, filter_n_atoms)

        # Construct PyTorch dataloaders from datasets
        preprocess = PreprocessQM9(load_charges=cfg.include_charges)
        dataloaders = {split: DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=args.shuffle if (split == 'train') else False,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         collate_fn=preprocess.collate_fn)
                       for split, dataset in datasets.items()}
    elif 'geom' in cfg.dataset:
        import build_geom_dataset
        from configs.datasets_config import get_dataset_info
        data_file = './data/geom/geom_drugs_30.npy'
        dataset_info = get_dataset_info(cfg.dataset, cfg.remove_h)

        # Retrieve QM9 dataloaders
        split_data = build_geom_dataset.load_split_data(data_file,
                                                        val_proportion=0.1,
                                                        test_proportion=0.1,
                                                        filter_size=cfg.filter_molecule_size)
        transform = build_geom_dataset.GeomDrugsTransform(dataset_info,
                                                          cfg.include_charges,
                                                          cfg.device,
                                                          cfg.sequential)
        dataloaders = {}
        for key, data_list in zip(['train', 'val', 'test'], split_data):
            dataset = build_geom_dataset.GeomDrugsDataset(data_list,
                                                          transform=transform)
            shuffle = (key == 'train') and not cfg.sequential

            # Sequential dataloading disabled for now.
            dataloaders[key] = build_geom_dataset.GeomDrugsDataLoader(
                sequential=cfg.sequential, dataset=dataset,
                batch_size=cfg.batch_size,
                shuffle=shuffle)
        del split_data
        charge_scale = None
    elif 'mol-grid' in cfg.dataset:
        import fff.data.molecular as molecular

        def mol_grid_collate(batch: Tuple[torch.Tensor]):
            pass

        batch_size = cfg.batch_size
        num_workers = cfg.num_workers
        datasets = molecular.make_2d_atom_grid_datasets(
            cfg.filter_n_atoms,
            **cfg.dataset_kwargs,
        )
        dataloaders = {
            split: DataLoader(
                PositionDataset(dataset),
                batch_size=batch_size,
                shuffle=split == 'train',
                num_workers=num_workers,
                pin_memory=True,
            )
            for split, dataset in zip(['train', 'valid', 'test'], datasets)
        }
        charge_scale = None
    else:
        raise ValueError(f'Unknown dataset {cfg.dataset}')

    return dataloaders, charge_scale


class PositionDataset(torch.utils.data.Dataset):
    def __init__(self, position_tensor_dataset: TensorDataset):
        self.position_tensor_dataset = position_tensor_dataset

    def __getitem__(self, index):
        position_tensor, = self.position_tensor_dataset[index]

        n_coords = position_tensor.shape[0]
        n_dim = 3
        n_atoms = n_coords // n_dim
        return {
            "positions": position_tensor.reshape(n_atoms, n_dim),
            "atom_mask": torch.ones(n_atoms),
            "edge_mask": torch.ones(n_atoms ** 2, 1),
            "one_hot": torch.ones(n_atoms, 1, dtype=torch.bool),
            "charges": torch.zeros(n_atoms, 1, dtype=torch.int),
        }

    def __len__(self):
        return len(self.position_tensor_dataset)


def filter_atoms(datasets, n_nodes):
    for key in datasets:
        dataset = datasets[key]
        idxs = dataset.data['num_atoms'] == n_nodes
        for key2 in dataset.data:
            dataset.data[key2] = dataset.data[key2][idxs]

        datasets[key].num_pts = dataset.data['one_hot'].size(0)
        datasets[key].perm = None
    return datasets
