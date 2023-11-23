import torch.nn
from torch import nn

from fff.base import ModelHParams
from .utils import batch_wrap, make_inn


class InjectiveFlowHParams(ModelHParams):
    inn_spec: list
    zero_init: bool = True


class InjectiveFlow(nn.Module):
    """
    This uses a INN to map from data to latent space and back.
    In the case that latent_dim < data_dim, the latent space is a subspace of the data space.
    For reverting, the latent space is padded with zeros.
    """
    hparams: InjectiveFlowHParams

    def __init__(self, hparams: dict | InjectiveFlowHParams):
        if not isinstance(hparams, InjectiveFlowHParams):
            hparams = InjectiveFlowHParams(**hparams)

        super().__init__()
        self.hparams = hparams
        self.model = self.build_model()

    @batch_wrap
    def _encode(self, x):
        return self.model(x, jac=False, rev=False)[0][..., :self.hparams.latent_dim]

    def _latent_encode(self, u):
        return u

    @batch_wrap
    def _decode(self, u):
        u = torch.cat([
            u,
            torch.zeros(*u.shape[:-1], self.model.shapes[0][0] - self.hparams.latent_dim, device=u.device, dtype=u.dtype)
        ], -1)
        return self.model(u, jac=False, rev=True)[0]

    def _latent_decode(self, z):
        return z

    def build_model(self) -> nn.Module:
        data_dim = self.hparams.data_dim
        return make_inn(self.hparams.inn_spec, data_dim, zero_init=self.hparams.zero_init)
