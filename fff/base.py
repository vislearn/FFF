from collections import namedtuple, defaultdict
from copy import deepcopy
from importlib import import_module
from math import prod, log10

import torch
from lightning_trainable import Trainable, TrainableHParams
from lightning_trainable.hparams import HParams
from lightning_trainable.trainable.trainable import auto_pin_memory, SkipBatch
from torch.distributions import Independent, Normal
from torch.nn import Sequential
from torch.utils.data import DataLoader, IterableDataset

import fff.data
from fff.distributions.multivariate_student_t import MultivariateStudentT
from fff.loss import volume_change_surrogate
from fff.model.utils import TrainWallClock
from fff.utils.func import compute_jacobian


class ModelHParams(HParams):
    data_dim: int
    cond_dim: int
    latent_dim: int | str = "data"


class FreeFormBaseHParams(TrainableHParams):
    latent_distribution: dict = dict(
        name="normal"
    )
    data_set: dict
    noise: float | list = 0.0
    track_train_time: bool = False

    models: list

    loss_weights: dict
    log_det_estimator: dict = dict(
        name="surrogate",
        hutchinson_samples=1
    )
    skip_val_nll: bool | int = False
    exact_train_nll_every: int | None = None

    warm_up_epochs: int | list = 0

    exact_chunk_size: None | int = None


LogProbResult = namedtuple("LogProbResult", ["z", "x1", "log_prob", "regularizations"])
VolumeChangeResult = namedtuple("VolumeChangeResult", ["out", "volume_change", "regularizations"])
ConditionedBatch = namedtuple("ConditionedBatch", [
    "x0", "x_noisy", "loss_weights", "condition", "dequantization_jac"
])


class FreeFormBase(Trainable):
    """
    This class abstracts the joint functionalities of free-form flows (FFF)
    and Free-form injective flows (FIF).
    """
    hparams: FreeFormBaseHParams

    def __init__(self, hparams: FreeFormBaseHParams | dict):
        train_data, val_data, test_data = fff.data.load_dataset(**hparams["data_set"])

        super().__init__(hparams, train_data=train_data, val_data=val_data, test_data=test_data)

        try:
            self._data_dim = train_data.data_dim
            self._data_cond_dim = train_data.cond_dim
            if self._data_cond_dim is None or self._data_dim is None:
                raise AttributeError
        except AttributeError:
            data_sample = train_data[0]
            self._data_dim = prod(data_sample[0].shape)
            if len(data_sample) == 1:
                self._data_cond_dim = 0
            else:
                if len(data_sample[1].shape) != 1:
                    raise NotImplementedError("More than one condition dimension is not supported.")
                self._data_cond_dim = data_sample[1].shape[0]

        # Build model
        self.models = build_model(self.hparams.models, self.data_dim, self.cond_dim)

        # Learnt latent distribution
        self.latents = {}
        default_latent = self.get_latent(self.device)
        if isinstance(default_latent, torch.nn.Module):
            self.learnt_latent = default_latent

    def train_dataloader(self) -> DataLoader | list[DataLoader]:
        """
        Configures the Train DataLoader for Lightning. Uses the dataset you passed as train_data.

        @return: The DataLoader Object.
        """
        if self.train_data is None:
            return []
        kwargs = {}
        try:
            kwargs["collate_fn"] = self.train_data.collate_fn
        except AttributeError:
            pass
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=not isinstance(self.train_data, IterableDataset),
            pin_memory=auto_pin_memory(self.hparams.pin_memory, self.hparams.accelerator),
            num_workers=self.hparams.num_workers,
            **kwargs,
        )

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        """
        Configures the Validation DataLoader for Lightning. Uses the dataset you passed as val_data.

        @return: The DataLoader Object.
        """
        if self.val_data is None:
            return []
        kwargs = {}
        try:
            kwargs["collate_fn"] = self.train_data.collate_fn
        except AttributeError:
            pass
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=auto_pin_memory(self.hparams.pin_memory, self.hparams.accelerator),
            num_workers=self.hparams.num_workers,
            **kwargs,
        )

    def test_dataloader(self) -> DataLoader | list[DataLoader]:
        """
        Configures the Test DataLoader for Lightning. Uses the dataset you passed as test_data.

        @return: The DataLoader Object.
        """
        if self.test_data is None:
            return []
        kwargs = {}
        try:
            kwargs["collate_fn"] = self.train_data.collate_fn
        except AttributeError:
            pass
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=auto_pin_memory(self.hparams.pin_memory, self.hparams.accelerator),
            num_workers=self.hparams.num_workers,
            **kwargs,
        )

    def get_latent(self, device):
        # Learnable distributions should just be moved to the right device, so that parameters are shared
        if self.latents and isinstance(next(iter(self.latents.values())), torch.nn.Module):
            return next(iter(self.latents.values())).to(device)

        if device not in self.latents:
            latent_hparams = deepcopy(self.hparams.latent_distribution)
            distribution_name = latent_hparams.pop("name")
            latent = self._make_latent(distribution_name, device, **latent_hparams)
            assert latent is not None, (f"Found None latent distribution for name {distribution_name}."
                                        f"This is likely due to an error in the code, not the config.")
            self.latents[device] = latent
        return self.latents[device]

    def _make_latent(self, name, device, **kwargs):
        if name == "normal":
            loc = torch.zeros(self.latent_dim, device=device)
            scale = torch.ones(self.latent_dim, device=device)

            return Independent(
                Normal(loc, scale), 1,
            )
        elif name == "student_t":
            df = self.hparams.latent_distribution["df"] * torch.ones(1, device=device)
            return MultivariateStudentT(df, self.latent_dim)
        else:
            raise ValueError(f"Unknown latent distribution: {name!r}")

    @property
    def latent_dim(self):
        return self.models[-1].hparams.latent_dim

    def is_conditional(self):
        return self._data_cond_dim != 0

    @property
    def cond_dim(self):
        soft_flow_cond_dim = 1 if isinstance(self.hparams.noise, list) else 0
        hp_aware_cond_dim = sum(
            1 if isinstance(weight, list) else 0
            for weight in self.hparams.loss_weights.values()
        )

        return self._data_cond_dim + soft_flow_cond_dim + hp_aware_cond_dim

    @property
    def data_dim(self):
        return self._data_dim

    def configure_callbacks(self):
        callbacks = super().configure_callbacks()
        if self.hparams.track_train_time:
            callbacks.append(TrainWallClock())
        return callbacks

    def encode(self, x, c, intermediate=False):
        if intermediate:
            outs = []
        for model in self.models:
            x = model.encode(x, c)
            if intermediate:
                outs.append(x)
        if intermediate:
            x = x, outs
        return x

    def decode(self, z, c, intermediate=False):
        if intermediate:
            outs = []
        for model in self.models[::-1]:
            z = model.decode(z, c)
            if intermediate:
                outs.append(z)
        if intermediate:
            z = z, outs
        return z

    def _encoder_jac(self, x, c, **kwargs):
        return compute_jacobian(
            x, self.encode, c,
            chunk_size=self.hparams.exact_chunk_size,
            **kwargs
        )

    def _decoder_jac(self, z, c, **kwargs):
        return compute_jacobian(
            z, self.decode, c,
            chunk_size=self.hparams.exact_chunk_size,
            **kwargs
        )

    def _encoder_volume_change(self, x, c, **kwargs) -> VolumeChangeResult:
        raise NotImplementedError

    def _decoder_volume_change(self, z, c, **kwargs) -> VolumeChangeResult:
        raise NotImplementedError

    def forward(self, x, c):
        return self.decode(self.encode(x, c), c)

    def _latent_log_prob(self, z, c):
        try:
            return self.get_latent(z.device).log_prob(z, c)
        except TypeError:
            return self.get_latent(z.device).log_prob(z)

    def sample(self, sample_shape, condition=None, latent_scale=1.0):
        """
        Sample via the decoder.
        """
        z = self.get_latent(self.device).sample(sample_shape) * latent_scale
        z = z.reshape(prod(sample_shape), *z.shape[len(sample_shape):])
        batch = [z]
        if condition is not None:
            batch.append(condition)
        c = self.apply_conditions(batch).condition
        x = self.decode(z, c)
        return x.reshape(sample_shape + x.shape[1:])

    def exact_log_prob(self, x, c=None, jacobian_target="decoder",
                       input_is_z=False, **kwargs) -> LogProbResult:
        metrics = {}

        if c is None and not self.is_conditional():
            c = torch.empty((x.shape[0], 0), device=x.device, dtype=x.dtype)

        if input_is_z:
            if jacobian_target != "decoder":
                raise NotImplementedError("Cannot compute encoder Jacobian for z input.")
            z = x
            vol_change_enc = None
        else:
            if jacobian_target in ["encoder", "both"]:
                volume_change_enc = self._encoder_volume_change(x, c, **kwargs)
                z = volume_change_enc.out
                vol_change_enc = volume_change_enc.volume_change

                metrics.update(volume_change_enc.regularizations)
                metrics["vol_change_encoder"] = vol_change_enc
            else:
                z = self.encode(x, c)
                vol_change_enc = None

        if jacobian_target in ["decoder", "both"]:
            volume_change_dec = self._decoder_volume_change(z, c, **kwargs)
            x1 = volume_change_dec.out
            vol_change_dec = -volume_change_dec.volume_change

            metrics.update(volume_change_dec.regularizations)
            metrics["vol_change_decoder"] = vol_change_dec
        else:
            x1 = self.decode(z, c)
            vol_change_dec = None

        if jacobian_target == "encoder":
            volume_change = vol_change_enc
        else:
            # If "both" is specified, we prefer the decoder
            volume_change = vol_change_dec

        latent_log_prob = self._latent_log_prob(z, c)

        # Add additional nll terms if requested
        for key, value in list(metrics.items()):
            if key.startswith("vol_change_"):
                metrics[key.replace("vol_change_", "nll_")] = -(latent_log_prob + value)

        return LogProbResult(
            z, x1, latent_log_prob + volume_change, metrics
        )

    def surrogate_log_prob(self, x, c, **kwargs) -> LogProbResult:
        # Then compute JtJ
        config = deepcopy(self.hparams.log_det_estimator)
        estimator_name = config.pop("name")
        assert estimator_name == "surrogate"

        encoder_intermediates = []
        decoder_intermediates = []

        def wrapped_encode(x):
            z, intermediates = self.encode(x, c, intermediate=True)
            encoder_intermediates.extend(intermediates)
            return z

        def wrapped_decode(z):
            x, intermediates = self.decode(z, c, intermediate=True)
            decoder_intermediates.extend(intermediates)
            return x

        out = volume_change_surrogate(
            x,
            wrapped_encode,
            wrapped_decode,
            **kwargs
        )
        volume_change = out.surrogate

        out.regularizations.update(self.intermediate_reconstructions(decoder_intermediates, encoder_intermediates))

        latent_prob = self._latent_log_prob(out.z, c)
        return LogProbResult(
            out.z, out.x1, latent_prob + volume_change, out.regularizations
        )

    def intermediate_reconstructions(self, decoder_intermediates, encoder_intermediates):
        regularizations = {}
        if len(decoder_intermediates) > 1:
            regularizations["intermediate_reconstruction_all"] = 0.0
        for idx, (a, b) in enumerate(zip(encoder_intermediates[:-1], decoder_intermediates[-1:0:-1])):
            if a.shape != b.shape:
                try:
                    b = b.view(a.shape)
                except Exception as e:
                    raise ValueError(f"Shapes do not match for intermediate reconstruction {idx}: {a.shape} vs {b.shape}") from e
            intermediate_loss = torch.sum((a - b).reshape(a.shape[0], -1) ** 2, -1)
            regularizations[f"intermediate_reconstruction_{idx}"] = intermediate_loss
            regularizations["intermediate_reconstruction_all"] += intermediate_loss
        return regularizations

    def _reconstruction_loss(self, a, b):
        return torch.sum((a - b).reshape(a.shape[0], -1) ** 2, -1)

    def compute_metrics(self, batch, batch_idx) -> dict:
        """
        Computes the metrics for the given batch.

        Rationale:
        - In training, we only compute the terms that are actually used in the loss function.
        - During validation, all possible terms and metrics are computed.

        :param batch:
        :param batch_idx:
        :return:
        """
        conditioned = self.apply_conditions(batch)
        loss_weights = conditioned.loss_weights
        x = conditioned.x_noisy
        c = conditioned.condition
        x0 = conditioned.x0
        deq_vol_change = conditioned.dequantization_jac

        loss_values = {}
        metrics = {}

        def check_keys(*keys):
            return any(
                (loss_key in loss_weights)
                and
                (
                    torch.any(loss_weights[loss_key] > 0)
                    if torch.is_tensor(loss_weights[loss_key]) else
                    loss_weights[loss_key] > 0
                )
                for loss_key in keys
            )

        # Empty until computed
        x1 = z = None

        # Negative log-likelihood
        if not self.training or (
                self.hparams.exact_train_nll_every is not None
                and batch_idx % self.hparams.exact_train_nll_every == 0
        ):
            key = "nll_exact" if self.training else "nll"
            # todo unreadable
            if self.training or (self.hparams.skip_val_nll is not True and (self.hparams.skip_val_nll is False or (
                    isinstance(self.hparams.skip_val_nll, int)
                    and batch_idx < self.hparams.skip_val_nll
            ))):
                with torch.no_grad():
                    log_prob_result = self.exact_log_prob(x=x, c=c, jacobian_target="both")
                z = log_prob_result.z
                x1 = log_prob_result.x1
                loss_values[key] = -log_prob_result.log_prob - deq_vol_change
                loss_values.update(log_prob_result.regularizations)
            else:
                loss_weights["nll"] = 0
        if self.training and check_keys("nll"):
            warm_up = self.hparams.warm_up_epochs
            if isinstance(warm_up, int):
                warm_up = warm_up, warm_up + 1
            nll_start, warm_up_end = warm_up
            if nll_start == 0:
                nll_warmup = 1
            else:
                nll_warmup = soft_heaviside(
                    self.current_epoch + batch_idx / len(
                        self.trainer.train_dataloader
                        if self.training else
                        self.trainer.val_dataloaders
                    ),
                    nll_start, warm_up_end
                )
            loss_weights["nll"] *= nll_warmup
            if check_keys("nll"):
                log_prob_result = self.surrogate_log_prob(x=x, c=c)
                z = log_prob_result.z
                x1 = log_prob_result.x1
                loss_values["nll"] = -log_prob_result.log_prob - deq_vol_change
                loss_values.update(log_prob_result.regularizations)

        # In case they were skipped above
        if z is None:
            z = self.encode(x, c)
        if x1 is None:
            x1 = self.decode(z, c)

        # Wasserstein distance of marginal to Gaussian
        with torch.no_grad():
            z_marginal = z.reshape(-1)
            z_gauss = torch.randn_like(z_marginal)

            z_marginal_sorted = z_marginal.sort().values
            z_gauss_sorted = z_gauss.sort().values

            metrics["z 1D-Wasserstein-1"] = (z_marginal_sorted - z_gauss_sorted).abs().mean()
            metrics["z std"] = torch.std(z_marginal)

        # Reconstruction
        if not self.training or check_keys("reconstruction", "noisy_reconstruction"):
            loss_values["reconstruction"] = self._reconstruction_loss(x0, x1)
            loss_values["noisy_reconstruction"] = self._reconstruction_loss(x, x1)

        # Cyclic consistency of latent code
        if not self.training or check_keys("z_reconstruction"):
            # Not reusing x1 from above, as it does not detach z
            z1 = self.encode(x1, c)
            loss_values["z_reconstruction"] = self._reconstruction_loss(z, z1)

        # Cyclic consistency of latent code -- gradient only to encoder
        if not self.training or check_keys("z_reconstruction_encoder"):
            # Not reusing x1 from above, as it does not detach z
            x1_detached = x1.detach()
            z1 = self.encode(x1_detached, c)
            loss_values["z_reconstruction_encoder"] = self._reconstruction_loss(z, z1)

        # Cyclic consistency of latent code sampled from Gauss
        if not self.training or check_keys("z_sample_reconstruction"):
            z_random = self.get_latent(z.device).sample((z.shape[0],))
            if isinstance(z_random, tuple):
                z_random, c_random = z_random
            else:
                c_random = c
            try:
                # Sanity checks might fail for random data
                z1_random = self.encode(self.decode(z_random, c_random), c_random)
                loss_values["z_sample_reconstruction"] = self._reconstruction_loss(z_random, z1_random)
            except:
                loss_values["z_sample_reconstruction"] = float("nan") * torch.ones(z_random.shape[0])

        # Reconstruction of Gauss with double std -- for invertibility
        if not self.training or check_keys("x_sample_reconstruction"):
            # As we only care about the reconstruction, can ignore noise scale
            x_random = self.get_latent(z.device).sample((z.shape[0],))
            if isinstance(x_random, tuple):
                x_random, c_random = x_random
            else:
                c_random = c
            try:
                # Sanity checks might fail for random data
                x1_random = self.decode(self.encode(x_random, c_random), c_random)
                loss_values["x_sample_reconstruction"] = self._reconstruction_loss(x_random, x1_random)
            except:
                loss_values["x_sample_reconstruction"] = float("nan") * torch.ones(x_random.shape[0])

        # Reconstruction of Gauss with double std -- for invertibility
        if not self.training or check_keys("shuffled_reconstruction"):
            # Make noise scale independent of applied noise, reconstruction should still be fine
            x_shuffled = x[torch.randperm(x.shape[0])]
            z_shuffled = self.encode(x_shuffled, c)
            x_shuffled1 = self.decode(z_shuffled, c)
            loss_values["shuffled_reconstruction"] = self._reconstruction_loss(x_shuffled, x_shuffled1)

        # Compute loss as weighted loss
        metrics["loss"] = sum(
            (weight * loss_values[key]).mean(-1)
            for key, weight in loss_weights.items()
            if check_keys(key) and (self.training or key in loss_values)
        )

        # Metrics are averaged, non-weighted loss_values
        invalid_losses = []
        for key, weight in loss_values.items():
            # One value per key
            if loss_values[key].shape != (x.shape[0],):
                invalid_losses.append(key)
            else:
                metrics[key] = loss_values[key].mean(-1)
        if len(invalid_losses) > 0:
            raise ValueError(f"Invalid loss shapes for {invalid_losses}")

        # Store loss weights
        if self.training:
            for key, weight in loss_weights.items():
                if not torch.is_tensor(weight):
                    weight = torch.tensor(weight)
                self.log(f"weights/{key}", weight.float().mean())

        # Check finite loss
        if not torch.isfinite(metrics["loss"]) and self.training:
            self.trainer.save_checkpoint("erroneous.ckpt")
            print(f"Encountered nan loss from: {metrics}!")
            raise SkipBatch

        return metrics

    def on_train_epoch_end(self) -> None:
        try:
            for key, value in self.val_data.compute_metrics(self).items():
                self.log(f"validation/{key}", value)
        except (AttributeError, TypeError):
            pass

    def on_fit_end(self) -> None:
        try:
            if self.hparams.data_set["name"].startswith("sbi_"):
                taskname = "_".join(self.hparams.data_set["name"].split("_")[1:])
                from fff.evaluate.c2st import c2st
                c2st_accuracy = c2st(self, taskname)
                self.logger.experiment.add_scalar("C2ST", c2st_accuracy, self.global_step)
        except Exception as e:
            # No need to give up a good run because of a plotting error
            print(e)
            pass

    def apply_conditions(self, batch) -> ConditionedBatch:
        x0 = batch[0]
        base_cond_shape = (x0.shape[0], 1)
        device = x0.device
        dtype = x0.dtype

        conds = []

        # Dataset condition
        if len(batch) != (2 if self.is_conditional() else 1):
            raise ValueError("You must pass a batch including conditions for each dataset condition")
        if len(batch) > 1:
            conds.append(batch[1])

        # SoftFlow
        noise_conds, x, dequantization_jac = self.dequantize(batch)
        conds.extend(noise_conds)

        # Loss weight aware
        loss_weights = defaultdict(float, self.hparams.loss_weights)
        for loss_key, loss_weight in self.hparams.loss_weights.items():
            if isinstance(loss_weight, list):
                min_weight, max_weight = loss_weight
                if not self.training:
                    # Per default, select the first value in the list
                    max_weight = min_weight
                weight_scale = rand_log_uniform(
                    min_weight, max_weight,
                    shape=base_cond_shape, device=device, dtype=dtype
                )
                loss_weights[loss_key] = (10 ** weight_scale).squeeze(1)
                conds.append(weight_scale)

        if len(conds) == 0:
            c = torch.empty((x.shape[0], 0), device=x.device, dtype=x.dtype)
        elif len(conds) == 1:
            # This is a hack to pass through the info dict from QM9
            c, = conds
        else:
            c = torch.cat(conds, -1)
        return ConditionedBatch(x0, x, loss_weights, c, dequantization_jac)

    def dequantize(self, batch):
        x0 = batch[0]
        base_cond_shape = (x0.shape[0], 1)
        device = x0.device
        dtype = x0.dtype

        noise = self.hparams.noise
        if isinstance(noise, list):
            min_noise, max_noise = noise
            if not self.training:
                max_noise = min_noise
            noise_scale = rand_log_uniform(
                max_noise, min_noise,
                shape=base_cond_shape, device=device, dtype=dtype
            )
            x = x0 + torch.randn_like(x0) * (10 ** noise_scale)
            noise_conds = [noise_scale]
        else:
            if noise > 0:
                x = x0 + torch.randn_like(x0) * noise
            else:
                x = x0
            noise_conds = []
        return noise_conds, x, torch.zeros(x0.shape[0], device=device, dtype=dtype)


def build_model(models, data_dim: int, cond_dim: int):
    if not isinstance(models[0], dict):
        return Sequential(*models)
    models = deepcopy(models)
    model = Sequential()
    for model_spec in models:
        module_name, class_name = model_spec.pop("name").rsplit(".", 1)
        model_spec["data_dim"] = data_dim
        model_spec["cond_dim"] = cond_dim
        if model_spec.get("latent_dim", "data") == "data":
            model_spec["latent_dim"] = data_dim
        model.append(
            getattr(import_module(module_name), class_name)(model_spec)
        )
        data_dim = model_spec["latent_dim"]
    return model


def soft_heaviside(pos, start, stop):
    return max(0., min(
        1.,
        (pos - start)
        /
        (stop - start)
    ))


def rand_log_uniform(vmin, vmax, shape, device, dtype):
    vmin, vmax = map(log10, [vmin, vmax])
    return torch.rand(
        shape, device=device, dtype=dtype
    ) * (vmin - vmax) + vmax


def wasserstein2_distance_gaussian_approximation(x1, x2):
    # Returns the squared 2-Wasserstein distance between the Gaussian approximation of two datasets x1 and x2
    # 1. Calculate mean and covariance of x1 and x2
    # 2. Use fact that tr( ( cov1^(1/2) cov2 cov1^(1/2) )^(1/2) ) = sum(eigvals( ( cov1^(1/2) cov2 cov1^(1/2) )^(1/2) ))
    # = sum(eigvals( cov1 cov2 )^(1/2))
    # 3. Return ||m1 - m2||^2 + tr( cov1 + cov2 - 2 ( cov1^(1/2) cov2 cov1^(1/2) )^(1/2) )
    m1 = x1.mean(0)
    m2 = x2.mean(0)
    cov1 = (x1 - m1[None]).T @ (x1 - m1[None]) / x1.shape[0]
    cov2 = (x2 - m2[None]).T @ (x2 - m2[None]) / x2.shape[0]
    cov_product = cov1 @ cov2
    eigenvalues_prod = torch.relu(torch.linalg.eigvals(cov_product).real)
    m_part = torch.sum((m1 - m2) ** 2)
    cov_part = torch.trace(cov1) + torch.trace(cov2) - 2 * torch.sum(torch.sqrt(eigenvalues_prod))
    return m_part + cov_part
