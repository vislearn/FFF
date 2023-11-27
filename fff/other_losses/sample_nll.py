import torch

from scipy.spatial import cKDTree
from collections import namedtuple

LogProbResult = namedtuple("LogProbResult", ["z", "x1", "log_prob", "regularizations"])


@torch.no_grad()
def get_sample_nll(model, x0, n_around_z=100, n_around_x=0, n_random_z=0, distance_cutoff=None):
    """
    Compute a sample-based estimate of the negative log-likelihood of x0 under the model.
    This may be more accurate than the decoder likelihood, since the encoder is not an exact inverse of the decoder.

    There are three ways to get sensible latent codes:
    1. Sample from a small region (std=0.01) around the latent code of x0 produced by the encoder.
    2. Sample from a small region (std=0.01) around x0 and encode.
    3. Sample from the latent distribution.

    :param model: Model to evaluate.
    :param x0: Data to evaluate. Shape: (batch_size, ...)
    :param n_around_z: Number of samples to draw around the latent code of x0.
    :param n_around_x: Number of samples to draw around x0.
    :param n_random_z: Number of samples to draw from the latent distribution.
    :param distance_cutoff: If provided, samples further away than this distance will be assigned inf nll.
    """
    x0 = x0.to("cpu")
    latent = model.get_latent(model.device)

    z_sampled = []
    x_sampled = []

    batch_size = x0.shape[0]

    if n_around_z > 0:
        x_cond = model.apply_conditions((x0,))
        z0 = model.encode(x_cond.x0, x_cond.condition)
        z_grid = z0.unsqueeze(0) + torch.randn(n_around_z, z0.shape[0], *z0.shape[1:]) * 0.01
        z_grid[0] = z0
        for batch in z_grid:
            if hasattr(model, "manifold"):
                z_grid = model.manifold.projection(batch)
            z_cond = model.apply_conditions((batch,))
            z_grid_samples_x = model.decode(z_cond.x0, z_cond.condition)
            z_sampled.append(z_grid.cpu())
            x_sampled.append(z_grid_samples_x.cpu())

    if n_around_x > 1:
        x_grid = x0.unsqueeze(0) + torch.randn(n_around_x, x0.shape[0], *x0.shape[1:]) * 0.01
        x_grid[0] = x0
        for batch in x_grid:
            if hasattr(model, "manifold"):
                batch = model.manifold.projection(batch)
            x_cond = model.apply_conditions((batch,))
            z_grid = model.encode(x_cond.x0, x_cond.condition)
            z_cond = model.apply_conditions((z_grid,))
            x_grid_samples_x = model.decode(z_cond.x0, z_cond.condition)
            z_sampled.append(z_grid.cpu())
            x_sampled.append(x_grid_samples_x.cpu())

    if n_random_z > 0:
        z_latent_sampled = latent.sample((n_random_z,))
        for batch in z_latent_sampled.split(batch_size):
            z_cond = model.apply_conditions((z_latent_sampled,))
            x_latent_sampled = model.decode(z_cond.x0, z_cond.condition)
            z_sampled.append(z_latent_sampled.cpu())
            x_sampled.append(x_latent_sampled.cpu())

    z_sampled = torch.cat(z_sampled, dim=0).reshape(-1, model._data_dim).cpu()
    x_sampled = torch.cat(x_sampled, dim=0).reshape(-1, model.latent_dim).cpu()

    assert len(x_sampled) > 0, "Need at least one sample to compute sample_nll"

    tree = cKDTree(x_sampled)
    _, indices = tree.query(x0.reshape(-1, model._data_dim), k=1)
    x_sampled_nearest = x_sampled[indices].reshape(-1, *model.manifold.shape)
    z_sampled_nearest = z_sampled[indices].reshape(-1, *model.manifold.shape)

    c1 = model.apply_conditions([z_sampled_nearest]).condition
    nll = -model.exact_log_prob(z_sampled_nearest, c1).log_prob

    try:
        manifold = model.manifold
        distances = manifold.metric.dist(x0, x_sampled_nearest)
    except AttributeError:
        distances = torch.norm(x0 - x_sampled_nearest, dim=-1)
    if distance_cutoff is not None:
        nll[distances > distance_cutoff] = torch.inf

    return nll, distances
