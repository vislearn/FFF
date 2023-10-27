import numpy as np
import torch
from FrEIA.utils import sum_except_batch
from fff.model.en_graph_utils.utils import remove_mean_with_mask


class PositionFeaturePrior:
    def __init__(self, n_dim, n_one_hot, n_charges, nodes_dist, device):
        self.n_dim = n_dim
        self.n_one_hot = n_one_hot
        self.n_charges = n_charges
        self.nodes_dist = nodes_dist
        self.device = device

    def log_prob(self, z_xh, condition=None):
        # Probability of picking this number of atoms
        if isinstance(condition, dict):
            node_mask = condition['node_mask']
            N = node_mask.squeeze(-1).sum(-1).long()
            log_pN = self.nodes_dist.log_prob(N)
        else:
            node_mask = None
            log_pN = 0

        z_x = z_xh[..., :self.n_dim]
        z_h = z_xh[..., self.n_dim:]
        if node_mask is None:
            node_mask = torch.ones((*z_x.shape[:-1], 1), device=z_x.device)
        assert len(node_mask.size()) == 3
        assert node_mask.size()[:2] == z_x.size()[:2]
        assert z_h.shape[-1] == self.n_one_hot + self.n_charges

        assert (z_x * (1 - node_mask)).sum() < 1e-8 and \
               (z_h * (1 - node_mask)).sum() < 1e-8, \
            'These variables should be properly masked.'

        log_pz_x = center_gravity_zero_gaussian_log_likelihood_with_mask(
            z_x, node_mask
        )

        log_pz_h = standard_gaussian_log_likelihood_with_mask(
            z_h, node_mask
        )

        log_pz = log_pz_x + log_pz_h + log_pN
        return log_pz

    def sample(self, shape, n_nodes=None):
        n_samples, = shape
        if n_nodes is None:
            n_nodes = self.nodes_dist.sample()

        node_mask = torch.ones((n_samples, n_nodes, 1), device=self.device)
        z_x = sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dim), device=self.device,
            node_mask=node_mask)
        z_h_one_hot = sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_one_hot), device=self.device,
            node_mask=node_mask)
        z_h_charges = sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_charges), device=self.device,
            node_mask=node_mask)

        return torch.cat([z_x, z_h_one_hot, z_h_charges], -1).reshape(n_samples, -1), {
            'positions': z_x,
            'node_mask': node_mask,
            'one_hot': z_h_one_hot,
            'charges': z_h_charges.squeeze(-1),
        }


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


def assert_mean_zero_with_mask(x, node_mask):
    assert_correctly_masked(x, node_mask)
    assert torch.sum(x, dim=1, keepdim=True).abs().max().item() < 1e-4, \
        'Mean is not zero'


def center_gravity_zero_gaussian_log_likelihood_with_mask(x, node_mask):
    assert len(x.size()) == 3
    B, N_embedded, D = x.size()
    assert_mean_zero_with_mask(x, node_mask)

    # r is invariant to a basis change in the relevant hyperplane, the masked
    # out values will have zero contribution.
    r2 = sum_except_batch(x.pow(2))

    # The relevant hyperplane is (N-1) * D dimensional.
    N = node_mask.squeeze(2).sum(1)  # N has shape [B]
    degrees_of_freedom = (N - 1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2 * np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    x_masked = x * node_mask

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected


def sample_center_gravity_zero_gaussian(size, device):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean(x)
    return x_projected


def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x


def standard_gaussian_log_likelihood_with_mask(x, node_mask):
    # Normalizing constant and logpx are computed:
    log_px_elementwise = -0.5 * x * x - 0.5 * np.log(2 * np.pi)
    log_px = sum_except_batch(log_px_elementwise * node_mask)
    return log_px


def sample_gaussian_with_mask(size, device, node_mask):
    x = torch.randn(size, device=device)
    x_masked = x * node_mask
    return x_masked
