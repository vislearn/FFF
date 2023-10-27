import functools
import pickle
import re
from functools import partial
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.distributions import Normal
from torch.utils.data import DataLoader

from torchvision.utils import make_grid
from tqdm.auto import tqdm
from yaml import dump

from lightning_trainable.launcher.utils import parse_config_dict
from fff.model.utils import guess_image_shape

from typing import List, Optional
from torch import Tensor


class SkipModel(BaseException):
    pass


def plot_density_2d(grad_to_enc_or_dec, model):
    grid = torch.meshgrid(*2 * [torch.linspace(-2.5, 2.5, 200)])
    batch = torch.stack(grid, -1).reshape(-1, 2)
    c = model.apply_conditions(batch).condition
    plt.title(grad_to_enc_or_dec)
    with torch.no_grad():
        batch_prob = model.log_prob(batch, c, grad_to_enc_or_dec=grad_to_enc_or_dec).exp()
    plt.contourf(*grid, batch_prob.reshape(grid[0].shape))
    plt.colorbar()


def get_n_samples_from_dataloader(data_loader: DataLoader, n_samples: int, condition=None):
    data = []
    collected_length = 0
    for batch in data_loader:
        if condition is not None:
            batch = tuple(
                t[torch.all(batch[1] == condition, 1)]
                for t in batch
            )
        data.append(batch)
        collected_length += batch[0].shape[0]
        if collected_length >= n_samples:
            break
    return tuple(
        torch.cat([tensor[i] for tensor in data])[:n_samples]
        for i in range(len(data[0]))
    )


def plot_reconstruction_grid(model, resolution=10):
    val_batch = get_n_samples_from_dataloader(model.val_dataloader(), 2048)
    val_batch = model.apply_conditions(val_batch).x_noisy
    plt.scatter(*val_batch.T, s=2)

    def noised_model(x):
        c = model.apply_conditions(x).condition
        return model(x, c)

    extent = -2.5, 2.5
    plot_grid(lambda x: x, *extent, color=".7", resolution=resolution)
    plot_grid(noised_model, *extent, sub_resolution=1000, resolution=resolution)
    plt.xlim(*extent)
    plt.ylim(*extent)


@torch.no_grad()
def plot_latent_reconstruction(model):
    val_data = torch.randn(2048, model.latent_dim)
    reconstruction = torch.cat([
        model(batch.to(model.device), model.apply_conditions(batch.to(model.device)).condition)
        for batch in val_data.split(model.hparams.batch_size)
    ])
    for i in range(val_data.shape[1]):
        diff = (reconstruction[:, i] - val_data[:, i]).abs()
        color = plt.get_cmap()(i / val_data.shape[1])
        if model.latent_dim == 0:
            val_data_sorted = val_data[:, i].sort()
            plt.plot(val_data_sorted.values, diff[val_data_sorted.indices], c=color)
        else:
            plt.scatter(val_data[:, i], diff,
                        s=1, color=color, alpha=max(.1, 1 / val_data.shape[1]))
    plt.yscale("log")
    plt.ylim(1e-3, 10)


def density1d(grad_to_enc_or_dec, model, condition=None):
    for condition in all_conditions(model, condition):
        sample_z = torch.randn(2 ** 14, model.latent_dim)
        if condition is None:
            conditions = []
        else:
            conditions = [torch.repeat_interleave(condition[None], len(sample_z), dim=0)]
        c = model.apply_conditions((sample_z, *conditions)).condition
        ordered_z = torch.sort(sample_z, 0)
        ordered_c = c[ordered_z.indices[:, 0]]
        with torch.no_grad():
            ordered_batch = model.decode(ordered_z.values, ordered_c)

        with torch.no_grad():
            sample_prob = model.log_prob(ordered_batch, c=ordered_c,
                                         grad_to_enc_or_dec=grad_to_enc_or_dec)[2].exp()
        return ordered_batch, sample_prob


def plot_manifold1d_density2d(grad_to_enc_or_dec, model, condition=None):
    for condition in all_conditions(model, condition):
        ordered_batch, sample_prob = density1d(grad_to_enc_or_dec, model, condition=condition)
        offset = 1 if grad_to_enc_or_dec == "encoder" else 0
        plt.scatter(*ordered_batch.T + offset, c=sample_prob, s=2, label=grad_to_enc_or_dec)
        plt.colorbar()
        plt.legend()


def plot_density_along_manifold1d(grad_to_enc_or_dec, model, condition=None):
    for condition in all_conditions(model, condition):
        ordered_batch, sample_prob = density1d(grad_to_enc_or_dec, model, condition=condition)
        cumpos = torch.cumsum((ordered_batch[1:] - ordered_batch[:-1]).norm(2, dim=-1), 0)
        plt.plot(cumpos, sample_prob[1:], label=grad_to_enc_or_dec)


@torch.no_grad()
def make_img_samples(model, temperatures=None, latent_dim: int = None, n_images: int = 10, random_state=None,
                     condition=None):
    if temperatures is None:
        temperatures = [0, .5, .8, 1]

    rng = np.random.default_rng(random_state)
    samples_x = []
    if latent_dim is None:
        latent_dim = model.latent_dim
    img_shape = guess_image_shape(model.data_dim)
    for condition in all_conditions(model, condition):
        for temperature in temperatures:
            sample_z = rng.normal(size=(n_images, latent_dim))
            sample_z = torch.from_numpy(sample_z).float().to(model.device) * temperature
            if condition is None:
                batch = (sample_z,)
            else:
                batch = (sample_z, repeat_condition(sample_z, condition).to(sample_z))
            c = model.apply_conditions(batch).condition
            samples_x.append(torch.cat([
                model.decode(z_batch, c_batch).cpu().reshape(-1, *img_shape)
                for z_batch, c_batch in zip(sample_z.split(model.hparams.batch_size), c.split(model.hparams.batch_size))
            ]))
    return torch.cat(samples_x)


@functools.wraps(make_img_samples)
def make_img_samples_grid(*args, n_images_per_row: int = 10, n_rows: int = 1, padding=0, **kwargs):
    samples = make_img_samples(*args, n_images=n_rows * n_images_per_row, **kwargs)
    return make_grid(samples, n_images_per_row, padding=padding)


@functools.wraps(make_img_samples_grid)
def plot_img_samples(model, *args, **kwargs):
    grid = make_img_samples_grid(model, *args, **kwargs)
    plt.imshow(grid.permute(1, 2, 0))


def compute_fid(model, temperature, n_samples: int, condition, device, fid, data_loader=None):
    if data_loader is None:
        data_loader = model.val_dataloader()

    img_shape = guess_image_shape(model.data_dim)
    is_grayscale = img_shape[0] == 1
    if n_samples is None:
        n_samples = model.hparams.batch_size
    val_samples = get_n_samples_from_dataloader(data_loader, n_samples, condition=condition)[0]
    for val_sample_batch in tqdm(val_samples.split(model.hparams.batch_size)):
        if is_grayscale:
            img_shape = list(img_shape)
            img_shape[0] = 3
            val_sample_batch = val_sample_batch.repeat(1, 3)
        fid.update(val_sample_batch.reshape(-1, *img_shape).to(device), True)

        samples = make_img_samples(model, temperatures=[temperature], n_images=val_sample_batch.shape[0],
                                   condition=condition)
        if is_grayscale:
            samples = samples.repeat(1, 3, 1, 1)
        fid.update(samples.reshape(-1, *img_shape).to(device), False)
    return fid.compute().cpu().item()


@torch.no_grad()
def plot_fid_by_tmp(model, temperatures=None, n_samples=None, condition=None, device="cuda", fid_feature=2048, data_loader=None):
    from torchmetrics.image import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=fid_feature, normalize=True).to(device)
    for condition in all_conditions(model, condition):
        if temperatures is None:
            temperatures = np.geomspace(.1, 10, 16)

        fids = []
        for temperature in tqdm(temperatures):
            fids.append(compute_fid(model, temperature, n_samples, condition, device, fid, data_loader))
            fid.reset()
        plt.plot(temperatures, fids)
        plt.xscale("log")
    return [temperatures, fids]


@torch.no_grad()
def make_img_reconstruction_grid(model, val_data=None, condition=None):
    n_images = 10
    if val_data is None:
        val_data = model.val_dataloader()
    image_rows = []
    img_shape = guess_image_shape(model.data_dim)
    for condition in all_conditions(model, condition):
        batch = get_n_samples_from_dataloader(val_data, n_images, condition)
        conditioned = model.apply_conditions(batch)
        images_reconstruction = model(
            conditioned.x_noisy.to(model.device),
            conditioned.condition.to(model.device)
        ).cpu()

        image_rows.append(condition.x_noisy.reshape(-1, *img_shape))
        image_rows.append(images_reconstruction.reshape(-1, *img_shape))

    return make_grid(torch.cat(image_rows), nrow=n_images)


@functools.wraps(make_img_reconstruction_grid)
def plot_img_reconstruction(*args, **kwargs):
    plt.imshow(make_img_reconstruction_grid(*args, **kwargs).permute(1, 2, 0))


def all_conditions(model, fixed_condition=None, collect_from_n_samples=1000):
    if fixed_condition is not None:
        return [fixed_condition]
    if not model.is_conditional():
        return [None]
    # Take the first n samples from the validation set and use their conditions
    _, val_conditions = get_n_samples_from_dataloader(model.val_dataloader(), collect_from_n_samples)
    return torch.unique(val_conditions, dim=0)


def repeat_condition(batch, condition):
    return torch.repeat_interleave(condition[None], len(batch), dim=0)


@torch.no_grad()
def plot_manifold(rectangular_flow, condition=None):
    eps = 1e-12

    for condition in all_conditions(rectangular_flow, condition):
        if condition is None:
            train_batch = rectangular_flow.train_data[:2048]
        else:
            all_x = rectangular_flow.train_data.tensors[0]
            x = all_x[torch.all(rectangular_flow.train_data.tensors[1] == condition, 1)][
                :2048]
            train_batch = (
                x,
                repeat_condition(x, condition)
            )
        conditioned = rectangular_flow.apply_conditions(train_batch)
        train_samples = conditioned.x_noisy
        c = conditioned.condition

        device = rectangular_flow.device

        batch_size = rectangular_flow.hparams.batch_size
        z_train = torch.cat([
            rectangular_flow.encode(batch.to(device), c_batch.to(device)).cpu()
            for batch, c_batch in zip(train_samples.split(batch_size), c.split(batch_size))
        ])
        x1_train = torch.cat([
            rectangular_flow.decode(batch.to(device), c_batch.to(device)).cpu()
            for batch, c_batch in zip(z_train.split(batch_size), c.split(batch_size))
        ])
        z_test = torch.randn_like(z_train)
        x1_test = torch.cat([
            rectangular_flow.decode(batch.to(device), c_batch.to(device)).cpu()
            for batch, c_batch in zip(z_test.split(batch_size), c.split(batch_size))
        ])

        plt.scatter(*train_samples.T, s=1, label="Original")
        plt.scatter(*x1_train.T, s=1, label="Reconstruction")
        plt.scatter(*x1_test.T, s=1, label="Samples")

        if z_train.shape[-1] == 1:
            u_eq = torch.linspace(-1 + eps, 1 - eps, 100, dtype=torch.float64)[:, None]
            z_eq = torch.erfinv(u_eq).float()
            if condition is None:
                batch = (z_eq,)
            else:
                batch = (z_eq, torch.repeat_interleave(condition[None, :], len(z_eq), 0))

            conditioned_eq = rectangular_flow.apply_conditions(batch)
            z_eq = conditioned_eq.x_noisy
            c_eq = conditioned_eq.condition
            x_eq = rectangular_flow.decode(z_eq.to(device), c_eq.to(device)).cpu()

            z_full = torch.cat([
                -10 ** torch.linspace(2, -2, 10_000)[:, None],
                10 ** torch.linspace(-2, 2, 10_000)[:, None]
            ])
            if condition is None:
                batch = (z_full,)
            else:
                batch = (z_full, torch.repeat_interleave(condition[None, :], len(z_full), 0))
            conditioned_eq = rectangular_flow.apply_conditions(batch)
            z_full = conditioned_eq.x_noisy
            c_full = conditioned_eq.condition
            x_full = rectangular_flow.decode(z_full.to(device), c_full.to(device)).cpu()

            plt.plot(*x_full.T, "-", lw=1, label="Manifold")
            plt.plot(*x_eq.T, ".", ms=2, label="Equally spaced")

    plt.xlim(-3, 3)
    plt.ylim(-3, 3)


def plot_latent_codes_2d_1d(model, condition=None, extent=2, cmap="twilight", data_cmap="twilight_shifted", normalize=True, repeat=False, fill=True, levels=20, **kwargs):
    grid_x = torch.stack(torch.meshgrid(2 * [torch.linspace(-extent, extent, 400)]), -1)
    for condition in all_conditions(model, condition):
        with torch.no_grad():
            x_flat = grid_x.reshape(-1, 2)
            if condition is None:
                batch = (x_flat,)
            else:
                batch = (x_flat, torch.repeat_interleave(condition[None, :], len(x_flat), 0))
            grid_c = model.apply_conditions(batch).condition
            grid_z = model.encode(x_flat, grid_c)
            # train_z = model.encode(train_batch)

        train_samples, *_ = get_n_samples_from_dataloader(model.val_dataloader(), 1000, condition=condition)
        plt.scatter(*train_samples.T, s=1, cmap=data_cmap)

        # plt.plot(*x_full.T.detach(), "w-", zorder=1)
        if repeat is not False:
            grid_z = grid_z % repeat
        if normalize:
            grid_z = torch.erf(grid_z)

        contour_fn = (plt.contourf if fill else plt.contour)
        contour_fn(grid_x[..., 0], grid_x[..., 1],
                   grid_z.reshape(grid_x.shape[:-1]),
                   levels=levels, zorder=0, cmap=cmap, **kwargs)

        plt.xlim(-extent, extent)
        plt.ylim(-extent, extent)


@torch.no_grad()
def plot_latent_codes(rectangular_flow, condition=None, n_samples=2 ** 13, vmin=-4, vmax=4):
    stds = []
    for condition in all_conditions(rectangular_flow, condition):
        device = rectangular_flow.device

        val_data = get_n_samples_from_dataloader(rectangular_flow.val_dataloader(), n_samples, condition=condition)
        conditioned = rectangular_flow.apply_conditions(val_data)
        z_train = torch.cat([
            rectangular_flow.encode(x_batch.to(device), c_batch.to(device)).cpu()
            for x_batch, c_batch in zip(
                conditioned.x_noisy.split(rectangular_flow.hparams.batch_size),
                conditioned.condition.split(rectangular_flow.hparams.batch_size)
            )
        ])

        z = torch.linspace(vmin, vmax, 100)
        plt.plot(z, Normal(0, 1).log_prob(z).exp())
        z_train_clamped = z_train.clamp(vmin, vmax).numpy()
        plt.hist(z_train_clamped, bins=np.linspace(vmin, vmax, 64),
                 histtype="step", density=True, alpha=.5)

        std = torch.std(z_train)
        plt.plot([0, std],
                 2 * [Normal(0, 1).log_prob(torch.zeros(1)).exp().item()],
                 "|-k",
                 label=f"$\sigma = {std:.1f}$")
        plt.legend()

        if std < .1:
            raise SkipModel

        stds.append(std)
    return stds


def ensure_list(value, dim):
    if isinstance(value, (int, float)):
        value = [value] * dim
    else:
        assert len(value) == dim
    return value


def build_mesh(pos_min, pos_max, resolution, dtype=None, device=None):
    pos_min = ensure_list(pos_min, 2)
    pos_max = ensure_list(pos_max, 2)
    resolution = ensure_list(resolution, 2)
    lin_spaces = [torch.linspace(a, b, r) for a, b, r in zip(pos_min, pos_max, resolution)]
    grid = [x.T for x in torch.meshgrid(lin_spaces)]
    pos = torch.stack(grid, 2).reshape(-1, 2)

    return *[g.to(device, dtype) for g in grid], pos.to(device, dtype)


@torch.no_grad()
def plot_grid(mapping, pos_min, pos_max, resolution=25,
              sub_resolution=0, color="black", linewidth=1, linestyle="-",
              pos_in_filter=None, device=None, dtype=None, **kwargs):
    for row_mode, this_resolution in zip([True, False],
                                         ensure_list(resolution, 2)):
        lines_resolution = [this_resolution,
                            this_resolution * (sub_resolution + 1)][::-1 if row_mode else 1]
        x, y, pos = build_mesh(pos_min, pos_max, lines_resolution,
                               device=device, dtype=dtype)
        if pos_in_filter is not None:
            pos[~pos_in_filter(pos)] = float("nan")
        pos_out = mapping(pos).cpu()
        pos_new = pos_out.reshape((*x.shape, -1)).numpy()
        if row_mode:
            segs = pos_new.transpose(1, 0, 2)
        else:
            segs = pos_new
        plt.plot(segs[:, :, 0], segs[:, :, 1], ls=linestyle, color=color,
                 lw=linewidth, **kwargs)


@torch.no_grad()
def plot_jjt_id(model, n_cols=4, n_rows=4, condition=None):
    for condition in all_conditions(model, condition):
        batch = get_n_samples_from_dataloader(model.val_dataloader(), n_cols * n_rows, condition)
        conditioned = model.apply_conditions(batch)

        Je = torch.func.vmap(torch.func.jacrev(model.encode))(conditioned.x_noisy, conditioned.condition)
        z0 = model.encode(conditioned.x_noisy, conditioned.condition)

        Jd = torch.func.vmap(torch.func.jacrev(model.decode))(z0, conditioned.condition)

        JJt = torch.bmm(Je, Jd)
        JJt_err = JJt - torch.eye(JJt.shape[-1])[None]
        JJt_grid = make_grid(JJt_err.unsqueeze(1), nrow=n_rows, value_range=(-1, 1))
        plt.imshow(JJt_grid[0], vmin=-1, vmax=1, cmap="bwr")
        plt.colorbar()


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', str(key))]
    return sorted(l, key=alphanum_key)


def expand_nested(dict_or_list, str_keys=False):
    """
    Recursively expand nested list/dictionary to flat dictionary.
    Resulting dictionary keys are tuples or strings of the path to the value.

    :param dict_or_list:
    :return:
    """
    if isinstance(dict_or_list, dict):
        items = dict_or_list.items()
    elif isinstance(dict_or_list, list):
        items = enumerate(dict_or_list)
    else:
        return {(): dict_or_list}

    result = {}
    for key, value in items:
        if isinstance(value, (dict, list)):
            result.update({(key,) + k: v for k, v in expand_nested(value).items()})
        else:
            result[(key,)] = value
    if str_keys:
        result = {"/".join(map(str, k)): v for k, v in result.items()}
    return result


def load_model(experiment_dir: Path | str, checkpoint_mode="last", overwrite_hparams: dict=None):
    from fff.model import FreeFormFlow
    experiment_dir = Path(experiment_dir)

    checkpoint_parent = experiment_dir / "checkpoints"
    if checkpoint_mode == "last":
        checkpoint = checkpoint_parent / "last.ckpt"
    else:
        checkpoints = natural_sort(checkpoint_parent.glob("*.ckpt"))
        if len(checkpoints) == 0:
            print(f"No checkpoints found in {experiment_dir}")
            return
        elif len(checkpoints) == 1:
            checkpoint = checkpoints[-1]
        else:
            if checkpoint_mode == "best":
                checkpoint = checkpoints[-2]
            else:
                for i, path in enumerate(checkpoints):
                    print(f"{i}: {path.name}")
                if isinstance(checkpoint_mode, int):
                    checkpoint = checkpoints[checkpoint_mode]
                else:
                    checkpoint = checkpoints[int(input("Select checkpoint: "))]
        print(checkpoint.name)
    if not checkpoint.is_file():
        print(f"{checkpoint} is not a file!")
        print(str(checkpoint))
        print(str(checkpoint.resolve()))
        return

    checkpoint_data = torch.load(checkpoint, map_location=torch.device('cpu'))
    hparams = checkpoint_data["hyper_parameters"]
    try:
        if overwrite_hparams is not None:
            parse_config_dict(overwrite_hparams, hparams)
        model = FreeFormFlow(hparams)
    except TypeError:
        if hparams["accumulate_batches"] is None:
            hparams["accumulate_batches"] = 1
        model = FreeFormFlow(hparams)
    try:
        model.load_state_dict(checkpoint_data["state_dict"])
    except RuntimeError:
        modified_state_dict = {
            f"models.0.{key}": value
            for key, value in checkpoint_data["state_dict"].items()
        }
        model.load_state_dict(modified_state_dict)
    model.eval()
    return model


def overview_figure(experiment_dir, hparam_keys=None, checkpoint_mode="last", separate_conditions=True, overwrite_hparams=None):
    model = load_model(experiment_dir, checkpoint_mode, overwrite_hparams=overwrite_hparams)
    if model is None:
        return

    plots = {}
    dim = model.data_dim
    if dim == 2:
        plots["Data & reconstruction"] = plot_manifold
    plots["Latent codes"] = plot_latent_codes
    if dim == model.latent_dim == 2:
        plots["Reconstruction grid"] = plot_reconstruction_grid
    if dim == 2 and model.latent_dim == 1:
        plots["Encoder density"] = partial(plot_manifold1d_density2d, "encoder")
        plots["Decoder density"] = partial(plot_manifold1d_density2d, "decoder")
        plots["Encoder density (along manifold)"] = partial(plot_density_along_manifold1d, "encoder")
        plots["Decoder density (along manifold)"] = partial(plot_density_along_manifold1d, "decoder")
    if dim == 2 and model.latent_dim == 2:
        plots["Encoder density"] = partial(plot_density_2d, "encoder")
        plots["Decoder density"] = partial(plot_density_2d, "decoder")
    if model.hparams.data_set["kind"] in ["mnist", "cifar10", "celeba"]:
        plots["Image samples"] = partial(plot_img_samples, temperatures=[.5, 1], n_rows=5, n_images_per_row=10)
        plots["Image reconstructions"] = plot_img_reconstruction
        if model.hparams.data_set["kind"] != "mnist":
            plots["FID by temperature"] = plot_fid_by_tmp
    if model.latent_dim <= 32 and model.data_dim <= 1024:
        plots["$J_eJ_d - I$"] = plot_jjt_id
    # plots["Marginal latent reconstructions"] = plot_latent_reconstruction

    if separate_conditions:
        conditions = all_conditions(model)
    else:
        conditions = [None]
    axss = plt.subplots(len(conditions), len(plots),
                        figsize=(len(plots) * 6, len(conditions) * 6),
                        constrained_layout=True, squeeze=False)[1]
    try:
        for axs, (name, plot_method) in zip(axss.T, plots.items()):
            for ax, condition in zip(axs, conditions):
                plt.sca(ax)
                plt.title(name)
                try:
                    result = plot_method(model, condition=condition)
                    if result is not None:
                        with open(experiment_dir / f"{name}.pkl", "wb") as f:
                            pickle.dump(result, f)
                except SkipModel:
                    raise
                except Exception as e:
                    print(f"Error in {name}: {e}")
    except SkipModel:
        pass

    plt.suptitle(experiment_dir.name)
    # plt.savefig(case_path / "overview.pdf")
    plt.show()
    if hparam_keys is not None:
        print(dump({
            key: model.hparams[key]
            for key in hparam_keys
            if key in model.hparams
        }, allow_unicode=True, default_flow_style=None))
    return model


def plot_convex_hull(data_2d: torch.Tensor):
    from scipy.spatial import ConvexHull
    hull = ConvexHull(data_2d.numpy())
    plt.plot(data_2d[:, 0], data_2d[:, 1], "k.")
    for simplex in hull.simplices:
        plt.plot(data_2d[simplex, 0], data_2d[simplex, 1], "r-")


def compute_decoder_jacobian(model, z, conditions=None):
    """
    Compute the Jacobian of the decoder with respect to the input z.
    Args:
        model (nn.Module): A model with implemented decode() method.
        z (Tensor): The input to the decode() method.
        conditions (Tensor): The conditions to the decode() method.
    Returns:
        Tensor: The Jacobian of the decoder with respect to the input z.
    """
    jacobians = []
    for condition in all_conditions(model, conditions):
        if condition is None:
            batch = (z,)
        else:
            batch = (z, repeat_condition(z, condition).to(z))
        conditioned = model.apply_conditions(batch)
        jacobian = torch.vmap(lambda z, c: torch.func.jacfwd(model.decode)(z, c))(
            conditioned.x0, conditioned.condition
        )
        jacobians.append(jacobian)
    return torch.cat(jacobians, dim=0)


def compute_decoder_jacobian_singular_values(model, z, conditions=None):
    """
    Compute the singular values of the Jacobian of the decoder with respect to the input z.
    Args:
        model (nn.Module): A model with implemented decode() method.
        z (Tensor): The input to the decode() method.
        conditions (Tensor): The conditions to the decode() method.
    Returns:
        Tensor: The singular values of the Jacobian of the decoder with respect to the input z.
    """
    jacobians = compute_decoder_jacobian(model, z, conditions)
    return torch.vmap(
        lambda jacobian: torch.linalg.svd(jacobian, full_matrices=False)[1]
    )(jacobians)


def plot_decoder_singular_value_spectrum(model, n_samples, temperature, plt_kwargs={}):
    """
    Plot the singular value spectrum of the Jacobian of the decoder with respect to the input z.
    Args:
        model (nn.Module): A model with implemented decode() method.
        n_samples (int): The number of samples to use for the estimation.
        temperature (float): The temperature of the normal distribution used for sampling.
        plt_kwargs (dict): Keyword arguments passed to plt.plot().
    Returns:
        None
    """
    z_batch = torch.normal(
        torch.zeros(n_samples, model.latent_dim),
        temperature * torch.ones(n_samples, model.latent_dim),
    )
    with torch.no_grad():
        singular_value_spectra = compute_decoder_jacobian_singular_values(
            model, z_batch
        )

    mean = singular_value_spectra.mean(dim=0)
    std = singular_value_spectra.std(dim=0)
    s_numeration = torch.arange(1, model.latent_dim + 1)
    plt.plot(s_numeration, mean, **plt_kwargs)
    plt.fill_between(s_numeration, mean - std, mean + std, alpha=0.2)
    plt.xticks(s_numeration)


def compute_decoder_singular_values_ge_one(model, n_samples, temperature):
    """
    Computes the interception of the singular value spectrum with the horizontal line at one.
    Args:
        model (nn.Module): A model with implemented decode() method.
        n_samples (int): The number of samples to use for the estimation.
        temperature (float): The temperature of the normal distribution used for sampling.
    Returns:
        int: Interception of the singular value spectrum with the horizontal line at one
             rounded down to the nearest integer.
        int: Interception of the singular value spectrum minus its standard deviation with
             the horizontal line at one rounded down to the nearest integer.
        int: Interception of the singular value spectrum plus its standard deviation with
             the horizontal line at one rounded down to the nearest integer.
    """
    z_batch = torch.normal(
        torch.zeros(n_samples, model.latent_dim),
        temperature * torch.ones(n_samples, model.latent_dim),
    )
    with torch.no_grad():
        singular_value_spectra = compute_decoder_jacobian_singular_values(
            model, z_batch
        )

    mean = singular_value_spectra.mean(dim=0)
    std = singular_value_spectra.std(dim=0)
    singular_values_ge_one_mean = torch.sum(mean >= 1)
    singular_values_ge_one_mean_m_std = torch.sum((mean - std) >= 1)
    singular_values_ge_one_mean_p_std = torch.sum((mean + std) >= 1)

    return (
        singular_values_ge_one_mean,
        singular_values_ge_one_mean_m_std,
        singular_values_ge_one_mean_p_std,
    )
