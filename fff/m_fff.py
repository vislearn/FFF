from copy import deepcopy
from math import prod
import torch

from fff.base import FreeFormBase, FreeFormBaseHParams, VolumeChangeResult, LogProbResult
from fff.loss import nll_surrogate


class ManifoldFreeFormFlowHParams(FreeFormBaseHParams):
    manifold_distance: bool = False


class ManifoldFreeFormFlow(FreeFormBase):
    """
    A ManifoldFreeFormFlow is a normalizing flow consisting of a pair of free-form
    encoder and decoder on a manifold.
    """

    hparams: ManifoldFreeFormFlowHParams

    def __init__(self, hparams: dict | ManifoldFreeFormFlowHParams):
        if not isinstance(hparams, ManifoldFreeFormFlowHParams):
            hparams = ManifoldFreeFormFlowHParams(**hparams)
        super().__init__(hparams)

        if not hasattr(self.train_data, "manifold"):
            raise ValueError("ManifoldFreeFormFlow requires a manifold to be specified in the train data.")

        if self.manifold is None:
            raise ValueError("ManifoldFreeFormFlow requires a manifold to be specified in the train data.")
        if self.manifold.projection is None:
            raise ValueError("ManifoldFreeFormFlow requires a projection to be specified in the manifold.")

    @property
    def manifold(self):
        return self.train_data.manifold

    def _make_latent(self, name, device, **kwargs):
        if not name.startswith("manifold"):
            raise ValueError("You have to use a manifold distribution when training on a manifold.")
        if name == "manifold-uniform":
            from fff.distributions.manifold_uniform import ManifoldUniformDistribution
            return ManifoldUniformDistribution(self.manifold, self.latent_dim, device=device)
        elif name == "manifold-von-mises-fisher":
            if "num_components" in kwargs.keys():
                n_modes = kwargs.pop("num_components")
                kwargs["n_modes"] = n_modes
            from fff.distributions.von_mises_fisher import VonMisesFisherMixtureDistribution
            with torch.inference_mode(False):
                return VonMisesFisherMixtureDistribution(self.manifold, **kwargs)
        else:
            raise ValueError(f"Unknown latent distribution: {name!r}")

    def encode(self, x, c, project=True, project_x=False):
        if project_x:
            x = self.manifold.projection(x)
        z = super().encode(x, c)
        if project:
            z = self.manifold.projection(z)
        return z

    def decode(self, z, c, project=True, project_z=False):
        if project_z:
            z = self.manifold.projection(z)
        x = super().decode(z, c)
        if project:
            x = self.manifold.projection(x)
        return x

    def _encoder_volume_change(self, x, c, **kwargs) -> VolumeChangeResult:
        z, jac_enc = self._encoder_jac(x, c, **kwargs)
        projected = project_jac_to_manifold(jac_enc, x, z, self.manifold)
        log_det = projected.slogdet()[1]
        return VolumeChangeResult(z, log_det, {})

    def _decoder_volume_change(self, z, c, **kwargs) -> VolumeChangeResult:
        x1, jac_dec = self._decoder_jac(z, c, **kwargs)
        projected = project_jac_to_manifold(jac_dec, z, x1, self.manifold)
        log_det = projected.slogdet()[1]
        return VolumeChangeResult(x1, log_det, {})

    def surrogate_log_prob(self, x, c, **kwargs) -> LogProbResult:
        # Then compute JtJ
        config = deepcopy(self.hparams.log_det_estimator)
        estimator_name = config.pop("name")
        assert estimator_name == "surrogate"

        out = nll_surrogate(
            x,
            lambda _x: self.encode(_x, c, project=False),
            lambda z: self.decode(z, c, project=False),
            manifold=self.manifold,
            **kwargs
        )
        volume_change = out.surrogate

        latent_prob = self._latent_log_prob(out.z, c)
        return LogProbResult(
            out.z, out.x1, latent_prob + volume_change, out.regularizations
        )

    def dequantize(self, batch):
        dequantize_out = super().dequantize(batch)
        noisy_data = dequantize_out[1]
        if noisy_data is not batch[0]:
            x_projected = self.manifold.projection(noisy_data)
            dequantize_out = (dequantize_out[0], x_projected, *dequantize_out[2:])
        return dequantize_out

    def _reconstruction_loss(self, a, b):
        if self.hparams.manifold_distance:
            return self.manifold.metric.squared_dist(a, b)
        return super()._reconstruction_loss(a, b)


def project_jac_to_manifold(jac, x_in, x_out, manifold):
    bases = []
    # Compute a basis each for x, z, and x1
    for pos in [x_in, x_out]:
        bs, dim = pos.shape[0], prod(pos.shape[1:])
        # This is a (bs, dim, manifold_dim) tensor
        tangents = torch.stack([
            random_tangent_vec(manifold, pos, n_samples=bs).reshape(bs, dim)
            for _ in range(manifold.dim)
        ], -1)
        basis, _ = torch.linalg.qr(tangents)
        bases.append(basis)
    x_in_basis, x_out_basis = bases

    # Project the Jacobian after reshaping to bs x out_dim x in_dim
    x_in_dim = prod(x_in.shape[1:])
    x_out_dim = prod(x_out.shape[1:])
    jac_vec = jac.reshape(jac.shape[0], x_out_dim, x_in_dim)
    return torch.bmm(
        torch.bmm(x_out_basis.transpose(-1, -2), jac_vec),
        x_in_basis
    )


def random_tangent_vec(manifold, base_point, n_samples):
    """Generate random tangent vec.

    Copied from geomstats with the right device handling.
    Also do not squeeze the batch dimension.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    base_point :  array-like, shape={[n_samples, *point_shape]}
        Point.

    Returns
    -------
    tangent_vec : array-like, shape=[..., *point_shape]
        Tangent vec at base point.
    """
    if (
            n_samples > 1
            and base_point.ndim > len(manifold.shape)
            and n_samples != len(base_point)
    ):
        raise ValueError(
            "The number of base points must be the same as the "
            "number of samples, when the number of base points is different from 1."
        )
    return manifold.to_tangent(
        torch.randn(size=(n_samples,) + manifold.shape, device=base_point.device), base_point
    )
