import os
from collections import Counter
import numpy as np
import torch
import pandas
import h5py
from torch.utils.data import TensorDataset

from fff.data.utils import TrainValTest

# download tabular data with the command (run from the repo main directory):
# mkdir -p data/ && wget -O - https://zenodo.org/record/1161203/files/data.tar.gz | tar --strip-components=1 -C data/ -xvzf - data/{gas,hepmass,miniboone,power}


# Modified from https://github.com/layer6ai-labs/rectangular-flows/blob/2caa5d01992b60ac8696123d4a7dacf0010d85a7/cif/datasets/tabular.py

def normalize_raw_data(data, mu, s):
    return (data - mu)/s


def make_tabular_train_valid_split(data, frac):
    n_valid = int(frac*data.shape[0])
    valid_data = data[-n_valid:]
    train_data = data[0:-n_valid]
    return train_data, valid_data


def make_tabular_train_valid_test_split(data, frac):
    n_test = int(frac*data.shape[0])
    test_data = data[-n_test:]
    data = data[0:-n_test]

    train_data, valid_data = make_tabular_train_valid_split(data, frac)
    return train_data, valid_data, test_data


def get_miniboone_raw(data_root):
    data = np.load(os.path.join(data_root, "miniboone/data.npy"))

    train_raw, valid_raw, test_raw = make_tabular_train_valid_test_split(data, 0.1)

    data_stack = np.vstack((train_raw, valid_raw))
    mu = data_stack.mean(axis=0)
    s = data_stack.std(axis=0)

    train_raw = normalize_raw_data(train_raw, mu, s)
    valid_raw = normalize_raw_data(valid_raw, mu, s)
    test_raw = normalize_raw_data(test_raw, mu, s)

    return train_raw, valid_raw, test_raw


def get_gas_raw(data_root):

    def get_gas_correlation_numbers(data):
        C = data.corr()
        A = C > 0.98
        B = A.to_numpy().sum(axis=1)
        return B

    try:
        data = pandas.read_pickle(os.path.join(data_root, "gas/ethylene_CO.pickle"))
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Please install an *older* pandas version than 2.X. "
                                  "If pandas 2.0 or newer is installed, convert ethylene_CO.pickle "
                                  "to the new format by loading with an older pandas version "
                                  "and saving with DataFrame.to_pickle(...). The resulting file is then "
                                  "compatible with newer pandas versions.")
    data.drop("Meth", axis=1, inplace=True)
    data.drop("Eth", axis=1, inplace=True)
    data.drop("Time", axis=1, inplace=True)

    B = get_gas_correlation_numbers(data)
    while np.any(B > 1):
        col_to_remove = np.where(B > 1)[0][0]
        col_name = data.columns[col_to_remove]
        data.drop(col_name, axis=1, inplace=True)
        B = get_gas_correlation_numbers(data)

    data = normalize_raw_data(data, data.mean(), data.std()).to_numpy()
    return make_tabular_train_valid_test_split(data, 0.1)


def get_hepmass_raw(data_root):
    train_data_path = os.path.join(data_root, "hepmass/1000_train.csv")
    test_data_path = os.path.join(data_root, "hepmass/1000_test.csv")

    train_raw = pandas.read_csv(filepath_or_buffer=train_data_path, index_col=False)
    test_raw = pandas.read_csv(filepath_or_buffer=test_data_path, index_col=False)

    train_raw = train_raw[train_raw[train_raw.columns[0]] == 1]
    train_raw = train_raw.drop(train_raw.columns[0], axis=1)

    test_raw = test_raw[test_raw[test_raw.columns[0]] == 1]
    test_raw = test_raw.drop(test_raw.columns[0], axis=1)
    test_raw = test_raw.drop(test_raw.columns[-1], axis=1)

    mu = train_raw.mean()
    s = train_raw.std()
    train_raw = normalize_raw_data(train_raw, mu, s).to_numpy()
    test_raw = normalize_raw_data(test_raw, mu, s).to_numpy()

    i = 0
    features_to_remove = []
    for feature in train_raw.T:
        c = Counter(feature)
        max_count = np.array([v for k, v in sorted(c.items())])[0]
        if max_count > 5:
            features_to_remove.append(i)
        i += 1
    train_raw = train_raw[:, np.array([i for i in range(train_raw.shape[1]) if i not in features_to_remove])]
    test_raw = test_raw[:, np.array([i for i in range(test_raw.shape[1]) if i not in features_to_remove])]

    train_raw, valid_raw = make_tabular_train_valid_split(train_raw, 0.1)
    return train_raw, valid_raw, test_raw


def get_power_raw(data_root):
    data = np.load(os.path.join(data_root, "power/data.npy"))
    np.random.shuffle(data)
    n = data.shape[0]

    data = np.delete(data, 3, axis=1)
    data = np.delete(data, 1, axis=1)

    gap_noise = 0.001*np.random.rand(n, 1)
    voltage_noise = 0.01*np.random.rand(n, 1)
    sm_noise = np.random.rand(n, 3)
    time_noise = np.zeros((n, 1))

    noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
    data = data + noise

    train_raw, valid_raw, test_raw = make_tabular_train_valid_test_split(data, 0.1)

    train_and_valid = np.vstack((train_raw, valid_raw))
    mu = train_and_valid.mean(axis=0)
    s = train_and_valid.std(axis=0)

    train_raw = normalize_raw_data(train_raw, mu, s)
    valid_raw = normalize_raw_data(valid_raw, mu, s)
    test_raw = normalize_raw_data(test_raw, mu, s)

    return train_raw, valid_raw, test_raw


def get_bsds300_raw(data_root):
    with h5py.File(os.path.join(data_root, "BSDS300", "BSDS300.hdf5"), "r") as f:
        train_raw = f["train"][()]
        valid_raw = f["validation"][()]
        test_raw = f["test"][()]
    return train_raw, valid_raw, test_raw


def get_raw_tabular_datasets(name, data_root):
    if name == "miniboone":
        data_fn = get_miniboone_raw
    elif name == "gas":
        data_fn = get_gas_raw
    elif name == "hepmass":
        data_fn = get_hepmass_raw
    elif name == "power":
        data_fn = get_power_raw
    elif name == "bsds300":
        data_fn = get_bsds300_raw
    else:
        raise NotImplementedError

    return data_fn(data_root)


def get_tabular_datasets(name, root) -> TrainValTest:
    train_raw, valid_raw, test_raw = get_raw_tabular_datasets(name, root)

    train_dset = TensorDataset(torch.tensor(train_raw, dtype=torch.get_default_dtype()))
    valid_dset = TensorDataset(torch.tensor(valid_raw, dtype=torch.get_default_dtype()))
    test_dset = TensorDataset(torch.tensor(test_raw, dtype=torch.get_default_dtype()))

    return train_dset, valid_dset, test_dset
