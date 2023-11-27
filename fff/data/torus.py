import os

import numpy as np
import pandas as pd
import torch
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.product_manifold import ProductManifold
from torch import Tensor
from torch.utils.data import TensorDataset

from .utils import split_dataset
from .manifold import ManifoldDataset


def embed_angle_in_2d(angle: Tensor) -> Tensor:
    x, y = torch.cos(angle), torch.sin(angle)
    return torch.stack([x, y], dim=-1)


def get_torus_protein_dataset(root: str = "./fff/data", subtype: str = None, seed: int = np.random.seed()):
    if subtype is None:
        subtype = "General"
    file_path = os.path.join(root, "raw_data", "torus", "protein.tsv")

    raw_data = pd.read_csv(file_path, delimiter="\t", header=None)
    raw_data.columns = ["name", "phi", "psi", "subtype"]
    subtype_data = raw_data[raw_data["subtype"] == subtype]

    phi_embedding = embed_angle_in_2d(
        torch.tensor(subtype_data["phi"].values) * 2 * torch.pi / 360
    ).unsqueeze(-1)
    psi_embedding = embed_angle_in_2d(
        torch.tensor(subtype_data["psi"].values) * 2 * torch.pi / 360
    ).unsqueeze(-1)
    data = torch.cat([phi_embedding, psi_embedding], dim=-1).transpose(1, 2)

    train_data, val_data, test_data = split_dataset(data, seed=seed)
    
    manifold = ProductManifold([Hypersphere(1), Hypersphere(1)])
    return (
        ManifoldDataset(TensorDataset(train_data), manifold),
        ManifoldDataset(TensorDataset(val_data), manifold),
        ManifoldDataset(TensorDataset(test_data), manifold),
    )


def get_torus_rna_dataset(root: str = "./fff/data", seed: int = np.random.seed()):
    file_path = os.path.join(root, "raw_data", "torus", "rna.tsv")
    
    raw_data = pd.read_csv(file_path, delimiter="\t", header=None)
    raw_data.columns = ["pdb_id", "resname", "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]

    embeddings = []
    for angle in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]:
        raw_data[angle] = raw_data[angle] * 2 * torch.pi / 360
        embeddings.append(embed_angle_in_2d(torch.tensor(raw_data[angle].values)).unsqueeze(-1))
    data = torch.cat(embeddings, dim=-1).transpose(1, 2)

    train_data, val_data, test_data = split_dataset(data, seed=seed)
    
    manifold = ProductManifold([Hypersphere(1) for _ in range(7)])
    return (
        ManifoldDataset(TensorDataset(train_data), manifold),
        ManifoldDataset(TensorDataset(val_data), manifold),
        ManifoldDataset(TensorDataset(test_data), manifold),
    )