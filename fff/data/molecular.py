import os

import numpy as np
import torch
from math import ceil, sqrt

from fff.data.qm9.data.utils import initialize_datasets
from fff.data.qm9.dataset import filter_atoms
from fff.model.en_graph_utils.utils import remove_mean
from torch.utils.data import TensorDataset


def make_2d_atom_grid_with_noise(n_atoms: int, N, center, shuffled,
                                 random_state: int, noise_std):
    index = np.arange(n_atoms)
    base_count = ceil(sqrt(n_atoms))

    x_index = index % base_count
    y_index = index // base_count

    positions = np.stack([
        x_index, y_index, np.zeros(n_atoms)
    ], -1)

    # Initialize the random number generator
    rng = np.random.default_rng(random_state)

    # Generate N shuffled versions of the atom positions
    if shuffled:
        shuffled_positions = np.array([rng.permutation(positions) for _ in range(N)])
    else:
        shuffled_positions = np.array([positions for _ in range(N)])

    # Add Gaussian noise to the shuffled positions
    noise = rng.normal(0, noise_std, shuffled_positions.shape)
    shuffled_positions_with_noise = torch.from_numpy(shuffled_positions + noise).float()

    if center:
        shuffled_positions_with_noise = remove_mean(shuffled_positions_with_noise)

    return shuffled_positions_with_noise


def make_2d_atom_grid_datasets(
        n_atoms: int, N_train=100_000, N_val=1_000, N_test=5_000,
        random_state: int = 12479,
        noise_std=0.1, shuffled=True, center=True
):
    N = N_train + N_val + N_test
    shuffled_positions_tensor = make_2d_atom_grid_with_noise(
        n_atoms, N,
        random_state=random_state,
        noise_std=noise_std, shuffled=shuffled, center=center
    ).reshape(N, -1)

    # Split into train, validation, and test sets
    train_data = shuffled_positions_tensor[:N_train]
    val_data = shuffled_positions_tensor[N_train:N_train + N_val]
    test_data = shuffled_positions_tensor[N_train + N_val:]

    # Create TensorDataset for each split
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)
    test_dataset = TensorDataset(test_data)

    return train_dataset, val_dataset, test_dataset


def load_qm9_dataset(
        root,
        filter_n_atoms=None, subtract_thermo=True, remove_h=False,
        force_download=False, include_charges=True
):
    datasets, num_species, charge_scale = initialize_datasets(
        datadir=root, dataset="qm9",
        subtract_thermo=subtract_thermo,
        force_download=force_download,
        remove_h=remove_h,
        include_charges=include_charges
    )
    qm9_to_eV = {
        'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114,
        'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114, 'lumo': 27.2114
    }

    for dataset in datasets.values():
        dataset.convert_units(qm9_to_eV)

    if filter_n_atoms is not None:
        print("Retrieving molecules with only %d atoms" % filter_n_atoms)
        datasets = filter_atoms(datasets, filter_n_atoms)

    return datasets["train"], datasets["valid"], datasets["test"]


def load_dw4_dataset(
        root,
        dim=8, n_particles=4,
        a=0.9, b=-4, c=0, offset=4,
        val_subset=10_000
):
    if dim != 8 or n_particles != 4 or a != 0.9 or b != -4 or c != 0 or offset != 4:
        raise ValueError("Parameters must match the sampled data.")

    # from bgflow import MultiDoubleWellPotential
    # target = MultiDoubleWellPotential(dim, n_particles, a, b, c, offset, two_event_dims=False)

    n_dimensions = dim // n_particles
    dw4_data = np.load(f"{root}/dw4-dataidx.npy", allow_pickle=True)
    atom_positions = dw4_data[0].reshape(-1, n_particles, n_dimensions)

    all_data = remove_mean(atom_positions)
    idx = dw4_data[1]
    train_data = all_data[idx[:100_000]]
    # Take a part of the validation data only
    val_data = all_data[idx[100_000:500_000]][:val_subset]
    test_data = all_data[idx[-500_000:]]

    return TensorDataset(train_data), TensorDataset(val_data), TensorDataset(test_data)


def load_lj13_dataset(
        root, random_seed=3412
):
    # first define system dimensionality and a target energy/distribution
    dim = 39
    n_particles = 13
    n_dimensions = dim // n_particles

    data_load = np.load(f"{root}/all_data_LJ13-2.npy", allow_pickle=True)
    atom_positions = torch.from_numpy(data_load.reshape(-1, n_particles, n_dimensions))
    idx = np.random.default_rng(random_seed).permutation(len(atom_positions))

    all_data = remove_mean(atom_positions)
    train_data = all_data[idx[:100_000]]
    # Take a part of the validation data only
    val_data = all_data[idx[100_000:500_000]][:10_000]
    test_data = all_data[idx[-500_000:]]

    return TensorDataset(train_data), TensorDataset(val_data), TensorDataset(test_data)


def load_lj55_dataset(
        root, random_seed=3412
):
    # first define system dimensionality and a target energy/distribution
    dim = 165
    n_particles = 55
    n_dimensions = dim // n_particles

    data_load = np.load(f"{root}/all_data_LJ55.npy", allow_pickle=True)
    atom_positions = torch.from_numpy(data_load.reshape(-1, n_particles, n_dimensions))
    idx = np.random.default_rng(random_seed).permutation(len(atom_positions))

    all_data = remove_mean(atom_positions)
    train_data = all_data[idx[:100_000]]
    # Take a part of the validation data only
    val_data = all_data[idx[100_000:500_000]][:10_000]
    test_data = all_data[idx[-500_000:]]

    return TensorDataset(train_data), TensorDataset(val_data), TensorDataset(test_data)
