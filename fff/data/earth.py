import torch
from torch.utils.data import TensorDataset
import os
import requests
import numpy as np
import pandas as pd
from math import ceil
from fff.data.manifold import ManifoldDataset
from geomstats.geometry.hypersphere import Hypersphere

def lat_long_to_3d(x):
    lat = np.deg2rad(x[:, 0])  # Convert latitude from degrees to radians
    lon = np.deg2rad(x[:, 1])  # Convert longitude from degrees to radians
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], -1)

def download_file(url, path):
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful
    folder = "/".join(path.split("/")[:-1])
    if not os.path.isdir(folder):
        os.mkdir(folder)
    with open(path, 'wb') as file:
        file.write(response.content)


def split_dataset(data, seed=1241735):
    permuted = torch.from_numpy(np.random.default_rng(seed).permutation(data)).float()
    return permuted[:int(0.8 * len(permuted))], permuted[int(0.8 * len(permuted)):int(0.9 * len(permuted))], permuted[int(0.9 * len(permuted)):]


def get_earth_dataset(name, **kwargs):
    if name == "flood":
        return get_flood_dataset(**kwargs)
    elif name == "fire":
        return get_fire_dataset(**kwargs)
    elif name == "quakes":
        return get_quakes_dataset(**kwargs)
    elif name == "volcano":
        return get_volcano_dataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset {name}")

def prepare_dataset(file_path, github_url, skiprows, uniform_mixture=0.0):
    if not os.path.exists(file_path):
        download_file(github_url, file_path)

    lat_long_data = torch.from_numpy(pd.read_csv(file_path, skiprows=skiprows, header=None).to_numpy())
    raw_data = torch.from_numpy(lat_long_to_3d(lat_long_data))
    manifold = Hypersphere(2)

    if uniform_mixture > 0.0:
        seed = 1238423
        noise_off_sphere = torch.from_numpy(np.random.default_rng(seed).normal(0, 1, (ceil(len(raw_data)*uniform_mixture), 3))).float()
        noise_on_sphere = manifold.projection(noise_off_sphere)
        raw_data = torch.cat([raw_data, noise_on_sphere], 0)

    train_data, val_data, test_data = split_dataset(raw_data)

    return ManifoldDataset(TensorDataset(train_data), manifold), \
           ManifoldDataset(TensorDataset(val_data), manifold), \
           ManifoldDataset(TensorDataset(test_data), manifold)


def get_flood_dataset(root="./data", **kwargs):
    file_path = os.path.join(root, 'flood.csv')
    github_url = 'https://raw.githubusercontent.com/oxcsml/riemannian-score-sde/main/data/flood.csv'

    return prepare_dataset(file_path, github_url, skiprows=2, **kwargs)

def get_fire_dataset(root="./data", **kwargs):
    file_path = os.path.join(root, 'fire.csv')
    github_url = 'https://raw.githubusercontent.com/oxcsml/riemannian-score-sde/main/data/fire.csv'

    return prepare_dataset(file_path, github_url, skiprows=1, **kwargs)

def get_quakes_dataset(root="./data", **kwargs):
    file_path = os.path.join(root, 'quakes_all.csv')
    github_url = 'https://raw.githubusercontent.com/oxcsml/riemannian-score-sde/main/data/quakes_all.csv'

    return prepare_dataset(file_path, github_url, skiprows=4, **kwargs)

def get_volcano_dataset(root="./data", **kwargs):
    file_path = os.path.join(root, 'volerup.csv')
    github_url = 'https://raw.githubusercontent.com/oxcsml/riemannian-score-sde/main/data/volerup.csv'

    return prepare_dataset(file_path, github_url, skiprows=2, **kwargs)
