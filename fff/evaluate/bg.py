import warnings
from copy import deepcopy
from math import prod
from time import time

import bgflow
import matplotlib.pyplot as plt
import numpy as np
import torch
from bgflow import LennardJonesPotential
from bgflow.bg import sampling_efficiency
from bgflow.utils import (
    distance_vectors, distances_from_vectors, as_numpy
)
from torch.func import jacrev, vmap
from tqdm.auto import tqdm, trange

from fff.evaluate.utils import load_cache
from fff.utils.utils import batch_wrap


def _tgt_info(model):
    data_set_name = model.hparams.data_set["name"]
    if data_set_name == "dw4":
        a = 0.9
        b = -4
        c = 0
        offset = 4

        dim = 8
        n_particles = 4

        target = bgflow.MultiDoubleWellPotential(dim, n_particles, a, b, c, offset, two_event_dims=False)
    elif data_set_name.startswith("lj"):
        n_particles = int(data_set_name[2:])
        dim = n_particles * 3

        target = LennardJonesPotential(dim, n_particles, eps=1., rm=1, oscillator_scale=1, two_event_dims=False)
    else:
        raise ValueError(f"Unknown data set {data_set_name}")
    n_dimensions = dim // n_particles
    return dim, n_dimensions, n_particles, target


def bg_for_model(model):
    dim, n_dimensions, n_particles, target = _tgt_info(model)

    def flow(z, inverse=False, temperature=1.0):
        batch_size = z.shape[0]
        z = z.reshape(-1, n_particles, n_dimensions)
        conditioned = model.apply_conditions([z])

        if inverse:
            x = conditioned.x0
            with torch.no_grad():
                z = model.encode(x, conditioned.condition)
            with torch.enable_grad():
                z_iterative = torch.nn.Parameter(z.clone())
                optim = torch.optim.SGD([z_iterative], lr=.4)

                def compute_loss():
                    x1 = model.decode(z_iterative, conditioned.condition)
                    return torch.nn.functional.mse_loss(x1, x)

                for iter in range(0):
                    optim.zero_grad()
                    loss = compute_loss()
                    loss.backward()
                    with torch.no_grad():
                        if iter % 10 == 0:
                            x_dist = torch.linalg.norm(
                                (x - model.decode(z_iterative, conditioned.condition)).reshape(batch_size, -1),
                                dim=1
                            ).mean().item()
                            # print(f"x-distance {iter: 3d}: {x_dist:.2e}")
                            z_dist = torch.linalg.norm(
                                (z - z_iterative).reshape(batch_size, -1),
                                dim=1
                            ).mean().item()
                            # print(f"z-distance {iter: 3d}: {z_dist:.2e}")
                    optim.step()
            z = z_iterative.detach()
            factor = -1
        else:
            factor = 1

        with torch.inference_mode(False):
            with torch.no_grad():
                jac_dec, x1 = vmap(jacrev(double_output(batch_wrap(model.decode)), has_aux=True),
                                   chunk_size=model.hparams.exact_chunk_size)(z, conditioned.condition)
            n_x_dim = prod(x1.shape[1:])
            n_z_dim = prod(z.shape[1:])
            jac_dec = jac_dec.reshape(batch_size, n_z_dim, n_x_dim)

        if inverse:
            out = z
        else:
            out = x1
        return out.reshape(z.shape[0], -1), factor * compute_volume_change(jac_dec)[..., None]

    prior = bgflow.MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False).to(model.device)
    bg = bgflow.BoltzmannGenerator(prior, flow, target).to(model.device)
    return bg


@torch.no_grad()
def sample_boltzmann(model, ckpt_file, n_samples, temperature=1.0, force_update=None, normalize=False):
    if not normalize:
        warnings.warn("Sampling without normalization is not recommended (spuriously good ESS).")
    # DW parameters
    bg = bg_for_model(model)

    dim, n_dimensions, n_particles, target = _tgt_info(model)

    n_sample_batches = n_samples // model.hparams.batch_size

    cache_file = ckpt_file.parents[1] / f"cache_{ckpt_file.name}_{temperature}_{n_samples}.pt"

    if load_cache(ckpt_file, cache_file, force_update=force_update):
        bg_samples = torch.load(cache_file)
    else:
        latent_np = np.empty(shape=(0))
        samples_np = np.empty(shape=(0))
        log_w_np = np.empty(shape=(0))
        energies_np = np.empty(shape=(0))
        distances_x_np = np.empty(shape=(0))
        times_np = np.empty(shape=(0))

        with trange(n_sample_batches) as pbar:
            for _ in pbar:
                start = time()
                samples, latent, dlogp = bg.sample(model.hparams.batch_size, with_latent=True, with_dlogp=True, temperature=temperature)
                times_np = np.append(times_np, [(time() - start) / len(samples)])
                # latent = latent[0]
                log_weights = bg.log_weights_given_latent(
                    samples, latent, dlogp, normalize=normalize
                ).detach().cpu().numpy()
                latent_np = np.append(latent_np, latent.detach().cpu().numpy())
                samples_np = np.append(samples_np, samples.detach().cpu().numpy())
                distances_x = distances_from_vectors(
                    distance_vectors(samples.view(-1, n_particles, n_dimensions))).detach().cpu().numpy().reshape(-1)
                distances_x_np = np.append(distances_x_np, distances_x)

                log_w_np = np.append(log_w_np, log_weights)
                energies = target.energy(samples).detach().cpu().numpy()
                energies_np = np.append(energies_np, energies)

                bg_samples_tmp = {"log_w_np": log_w_np}
                current_ess = ess(bg_samples_tmp) * 100
                samples_per_s = 1 / np.mean(times_np)
                pbar.set_description(f"ESS: {current_ess :.2f}; #samples/s: {samples_per_s:.2f}")

        latent_np = latent_np.reshape(-1, dim)
        samples_np = samples_np.reshape(-1, dim)

        bg_samples = {
            "latent_np": latent_np,
            "samples_np": samples_np,
            "log_w_np": log_w_np,
            "energies_np": energies_np,
            "distances_x_np": distances_x_np,
            "times_np": times_np,
        }
        torch.save(bg_samples, cache_file)
    return bg, bg_samples


def ess(bg_samples):
    log_w_np = bg_samples["log_w_np"]
    return sampling_efficiency(torch.from_numpy(log_w_np)).item()


@torch.no_grad()
def nll(model, ckpt_file, bg, batch_limit=None, force_update=None, data="train"):
    cache_file = ckpt_file.parents[1] / f"nll_{ckpt_file.name}_{data}.txt"
    print(cache_file.exists())
    if load_cache(ckpt_file, cache_file, force_update=force_update):
        with cache_file.open("r") as f:
            nll = float(f.read())
    else:
        nlls = []
        if data == "train":
            data_loader = model.train_dataloader()
        elif data == "validation":
            data_loader = model.val_dataloader()
        elif data == "test":
            data_loader = model.test_dataloader()
        else:
            raise ValueError(f"Invalid data split {data!r}")
        with tqdm(data_loader) as pbar:
            for i, batch in enumerate(pbar):
                if batch_limit is not None and i == batch_limit:
                    break

                batch = batch[0].to(model.device)
                nlls.append(bg.energy(batch).mean())
                pbar.set_description(f"Avg NLL: {sum(nlls) / len(nlls):.2f}")
        nll = (sum(nlls) / len(nlls)).item()
        if batch_limit is None:
            with cache_file.open("w") as f:
                f.write(str(nll))

    return nll


def plot_energy_distributions(model, bg: bgflow.BoltzmannGenerator, bg_samples,
                              common_hist_kwargs=None, sample_hist_kwargs=None, bg_hist_kwargs=None):
    latent_np = bg_samples["latent_np"]
    samples_np = bg_samples["samples_np"]
    log_w_np = bg_samples["log_w_np"]
    energies_np = bg_samples["energies_np"]
    distances_x_np = bg_samples["distances_x_np"]

    target = bg._target

    data = model.test_data.tensors[0].reshape(len(model.test_data), -1)

    energies_data = target.energy(data).detach().cpu().numpy()
    energies_bg = energies_np

    energies_prior = target.energy(torch.from_numpy(latent_np)).detach().cpu().numpy()

    min_energy = min(energies_data.min(), energies_bg.min(), energies_prior.min())
    max_energy = max(energies_data.max(), energies_bg.max(), energies_prior.max())

    # plt.figure(figsize=(16,9))
    plt.figure()
    # plt.title(base_path.name)
    common_kwargs = dict(
        bins=100, density=True, range=(None, 0), alpha=0.4
    )
    if common_hist_kwargs is not None:
        common_kwargs.update(common_hist_kwargs)
    if common_kwargs["range"][0] is None:
        common_kwargs["range"] = (min_energy, common_kwargs["range"][1])
    if common_kwargs["range"][1] is None:
        common_kwargs["range"] = (common_kwargs["range"][0], max_energy)

    sample_kwargs = deepcopy(common_kwargs)
    if sample_hist_kwargs is not None:
        sample_kwargs.update(sample_hist_kwargs)
    plt.hist(energies_bg, **sample_kwargs, color="r", label="FFF samples")
    plt.hist(energies_data, **sample_kwargs, color="g", label="MCMC data (test)")

    bg_kwargs = deepcopy(common_kwargs)
    if bg_hist_kwargs is not None:
        bg_kwargs.update(bg_hist_kwargs)
    plt.hist(energies_bg.reshape(-1), **bg_kwargs, color="b", label="FFF re-weighted", weights=np.exp(log_w_np))

    plt.xlabel("Energy $u(x)$")
    plt.ylabel("Empirical density $p(u)$")
    plt.xticks()
    plt.yticks()
    plt.legend()


def plot_distance_distributions(model, bg_samples, show_prior=False):
    dim, n_dimensions, n_particles, target = _tgt_info(model)

    distances_x_np = bg_samples["distances_x_np"]

    data = model.test_data.tensors[0].reshape(len(model.test_data), -1)

    dists = distances_from_vectors(
        distance_vectors(data.view(-1, n_particles, dim // n_particles)))

    # plt.figure(figsize=(16,9))
    plt.figure()
    if show_prior:
        latent_np = bg_samples["latent_np"]
        dists_prior = distances_from_vectors(
            distance_vectors(torch.from_numpy(latent_np).view(-1, n_particles, dim // n_particles)))
        plt.hist(as_numpy(dists_prior.view(-1)[::2]), bins=100, density=True, alpha=0.5, label="prior")
    plt.hist(as_numpy(distances_x_np[::2]), bins=100, density=True, alpha=0.7, linewidth=1, label="BG")
    plt.hist(as_numpy(dists.view(-1)[::2]), bins=100, density=True, alpha=0.5, linewidth=1, label="data")
    plt.xlim(0, 7)
    plt.legend()
    plt.xlabel("Distance")
    plt.xticks()
    plt.yticks()
    # plt.show()
    # plt.close()
