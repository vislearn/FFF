from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
from fff.m_loss import reconstruction_loss
from fff.other_losses.sample_nll import get_sample_nll
from torch import Tensor, nn


@torch.no_grad()
def numeric_evaluation(
    model: nn.Module,
    mode: str = "val",
) -> pd.DataFrame:
    if mode == "val":
        x0 = model.val_data[:][0].to(model.device)
    elif mode == "test":
        x0 = model.test_data[:][0].to(model.device)
    else:
        raise ValueError(f"mode {mode} not known.")
    c = model.apply_conditions([x0]).condition.to(model.device)
    x1 = model.decode(model.encode(x0, c), c)

    nll_enc = -model.exact_log_prob(x0, c, jacobian_target="encoder").log_prob
    nll_dec = -model.exact_log_prob(x0, c, jacobian_target="decoder").log_prob
    recon = reconstruction_loss(x0, x1)
    sample_nll, sample_recon = get_sample_nll(model, x0)

    numeric_evaluation_df = pd.DataFrame(
        [
            {
                "nll_enc": nll_enc.mean().item(),
                "nll_dec": nll_dec.mean().item(),
                "reconstruction": recon.mean().item(),
                "sample_nll": sample_nll.mean().item(),
                "sample_recon": sample_recon.mean().item(),
            }
        ]
    ).T
    numeric_evaluation_df.columns = ["Value"]
    return numeric_evaluation_df


@torch.no_grad()
def convert_to_angles(x: Tensor, torus_dim: Optional[int] = None) -> Tensor:
    if torus_dim is None:
        torus_dim = x.shape[-2]
    assert x.shape[-1] == 2, "Last dim must be 2 (embedding dim of the 1-sphere)"
    x_angular = []
    for i in range(torus_dim):
        x_angular.append(torch.atan2(x[:, i, 0], x[:, i, 1]))
    return torch.stack(x_angular, dim=1)


@torch.no_grad()
def plot_model_log_densities(
    model: nn.Module,
    num_grid_points: int = 200,
    levels: int = 10,
    ax: Optional[plt.Axes] = None,
    fontsizes: dict = dict(TITLESIZE=24, LABELSIZE=20, TICKSIZE=16),
    use_reconstructed_grid: bool = False,
):
    pass

    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)

    range_angular = torch.linspace(-torch.pi, torch.pi, num_grid_points)
    phi_grid, psi_grid = torch.meshgrid(range_angular, range_angular)
    phi, psi = phi_grid.flatten().unsqueeze(-1), psi_grid.flatten().unsqueeze(-1)
    phi_embedded = model.manifold.factors[0].angle_to_extrinsic(phi)
    psi_embedded = model.manifold.factors[1].angle_to_extrinsic(psi)
    x0 = torch.stack([phi_embedded, psi_embedded], dim=1)

    c = model.apply_conditions([x0]).condition

    log_prob_result = model.exact_log_prob(
        x0.to(model.device), c.to(model.device), jacobian_target="decoder"
    )
    if use_reconstructed_grid:
        x1 = log_prob_result.x1.cpu()
        x1_angular = convert_to_angles(x1)
        phi, psi = x1_angular[..., 0], x1_angular[..., 1]
    else:
        x0_angular = convert_to_angles(x0)
        phi, psi = x0_angular[..., 0], x0_angular[..., 1]
    log_prob = log_prob_result.log_prob.cpu()

    contours = ax.tricontourf(phi, psi, log_prob, levels=levels, cmap="viridis")
    cbar = plt.colorbar(contours)
    cbar.set_label("Log density", fontsize=fontsizes.get("LABELSIZE"))
    cbar.ax.tick_params(labelsize=fontsizes.get("TICKSIZE"))

    ax.set_xlim(-torch.pi, torch.pi)
    ax.set_ylim(-torch.pi, torch.pi)

    ax.set_xticks(
        [-torch.pi, -torch.pi / 2, 0, torch.pi / 2, torch.pi],
        [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"],
    )
    ax.set_yticks(
        [-torch.pi, -torch.pi / 2, 0, torch.pi / 2, torch.pi],
        [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"],
    )
    ax.tick_params(labelsize=fontsizes.get("TICKSIZE"))

    ax.set_xlabel(r"$\Phi$", fontsize=fontsizes.get("LABELSIZE"))
    ax.set_ylabel(r"$\Psi$", fontsize=fontsizes.get("LABELSIZE"))

    test_data_angular = convert_to_angles(model.test_data[:][0])
    ax.scatter(
        test_data_angular[..., 0],
        test_data_angular[..., 1],
        s=1 / len(test_data_angular) * 2e3,
        c="black",
        alpha=0.1,
        label="validation data",
    )
