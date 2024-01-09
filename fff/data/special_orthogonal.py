import pickle
from pathlib import Path

import numpy as np
import torch
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.symmetric_matrices import SymmetricMatrices

from scipy.stats import special_ortho_group

from torch.utils.data import TensorDataset

from fff.data.manifold import ManifoldDataset, fix_device
from geomstats.geometry.special_orthogonal import _SpecialOrthogonalMatrices
import geomstats.backend as gs


def flip_determinant(matrix, det):
    """Change sign of the determinant if it is negative.

    For a batch of matrices, multiply the matrices which have negative
    determinant by a diagonal matrix :math:`diag(1,...,1,-1) from the right.
    This changes the sign of the last column of the matrix.

    Parameters
    ----------
    matrix : array-like, shape=[...,n ,m]
        Matrix to transform.

    det : array-like, shape=[...]
        Determinant of matrix, or any other scalar to use as threshold to
        determine whether to change the sign of the last column of matrix.

    Returns
    -------
    matrix_flipped : array-like, shape=[..., n, m]
        Matrix with the sign of last column changed if det < 0.
    """
    ones = gs.ones(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    reflection_vec = gs.concatenate([ones[:-1], gs.array(
        [-1.0]
    ).to(matrix)], axis=0)
    mask = gs.cast(det < 0, matrix.dtype)
    sign = mask[..., None] * reflection_vec + (1.0 - mask)[..., None] * ones
    return gs.einsum("...ij,...j->...ij", matrix, sign)


class DeviceSpecialOrthogonalMatrices(_SpecialOrthogonalMatrices):
    def projection(self, point):
        aux_mat = self._aux_submersion(point)
        inv_sqrt_mat = SymmetricMatrices.powerm(aux_mat, -1 / 2)
        rotation_mat = Matrices.mul(point, inv_sqrt_mat)
        det = gs.linalg.det(rotation_mat)
        # geomstats flip_determinant not compatible with torch.func.vmap
        return flip_determinant(rotation_mat, det)


def make_so_data(K: int = 16, n_dim: int = 3,
                 N_total: int = 100_000,
                 scale: int = 100,
                 stored_seed: int = None, root: str = None,
                 random_state=12479):
    manifold = DeviceSpecialOrthogonalMatrices(n_dim)
    manifold.projection = fix_device(manifold.projection)

    rng = np.random.default_rng(random_state)
    if stored_seed is None:
        # Copied from SpecialOrthogonal.random_uniform
        # random_point = rng.uniform(-1, 1, size=(K,) + manifold.shape) * np.pi
        # means = torch.from_numpy(manifold.regularize(random_point))
        means = torch.from_numpy(
            special_ortho_group(n_dim, seed=rng.integers(2 ** 32 - 1)).rvs(size=K)
        )
        precision = rng.gamma(shape=scale, scale=1, size=(K,))

        mean_select = torch.from_numpy(rng.integers(K, size=(N_total,)))  # randint(K, size=(N_total,))
        random_means = means[mean_select]

        # This is inconsistent with rsde, but they have a modified version of geomstats
        tangent_scale = 1 / torch.sqrt(torch.from_numpy(precision))
        ambiant_noise = rng.normal(size=(N_total, manifold.dim))
        samples = manifold.lie_algebra.matrix_representation(ambiant_noise, normed=True)
        tangent_vec = tangent_scale * manifold.compose(random_means, samples)
        samples = manifold.exp(tangent_vec, random_means)
    else:
        with open(Path(root) / f"so3_{K=}_seed={stored_seed}_n={N_total}.pkl", "rb") as f:
            stored = pickle.load(f)
        samples = torch.from_numpy(stored["samples"][:N_total])
        assert len(samples) == N_total

    N_val = 1_000
    N_test = 5_000
    N_train = N_total - N_val - N_test

    train_dataset = TensorDataset(samples[:N_train].float())
    val_dataset = TensorDataset(samples[N_train:N_train + N_val].float())
    test_dataset = TensorDataset(samples[N_train + N_val:].float())

    return [
        ManifoldDataset(ds, manifold=manifold)
        for ds in [train_dataset, val_dataset, test_dataset]
    ]
