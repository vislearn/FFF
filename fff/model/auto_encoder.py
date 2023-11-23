from collections import OrderedDict

import torch.nn
from torch import nn
from torch.nn import Module

from fff.base import ModelHParams
from fff.model.utils import make_dense


class FullyConnectedNetworkHParams(ModelHParams):
    layer_spec: list
    skip_connection: bool = False
    detached_latent: bool = False

    def __init__(self, **hparams):
        # Compatibility with old checkpoints
        if "latent_layer_spec" in hparams:
            assert len(hparams["latent_layer_spec"]) == 0
            del hparams["latent_layer_spec"]
        super().__init__(**hparams)


class SkipConnection(Module):
    def __init__(self, inner: Module, id_init=False):
        super().__init__()
        self.inner = inner
        if id_init:
            self.scale = torch.nn.Parameter(torch.zeros(1))
        else:
            self.scale = None

    def forward(self, x):
        out = self.inner(x)
        if self.scale is not None:
            out = out * self.scale
        return x[..., :out.shape[-1]] + out


class FullyConnectedNetwork(nn.Module):
    hparams: FullyConnectedNetworkHParams

    def __init__(self, hparams: dict | FullyConnectedNetworkHParams):
        if not isinstance(hparams, FullyConnectedNetworkHParams):
            hparams = FullyConnectedNetworkHParams(**hparams)

        super().__init__()
        self.hparams = hparams
        self.model = self.build_model()

        if self.hparams.latent_dim != self.hparams.data_dim and self.hparams.skip_connection:
            raise ValueError("Can only have skip connection if data_dim = latent_dim")

    def encode(self, x, c):
        return self.model.encoder(torch.cat([x, c], -1))

    def decode(self, z, c):
        return self.model.decoder(torch.cat([z, c], -1))

    def build_model(self) -> nn.Module:
        data_dim = self.hparams.data_dim
        cond_dim = self.hparams.cond_dim

        # Nonlinear projection
        widths = [
            data_dim,
            *self.hparams.layer_spec,
            self.hparams.latent_dim
        ]
        encoder = make_dense([
            widths[0] + cond_dim,
            *widths[1:]
        ], "silu")
        decoder = make_dense([
            widths[-1] + cond_dim,
            *widths[-2::-1]
        ], "silu")

        modules = OrderedDict(
            encoder=encoder,
            decoder=decoder
        )
        # Apply skip connections
        if self.hparams.skip_connection:
            new_modules = OrderedDict()
            for key, value in modules.items():
                new_modules[key] = SkipConnection(value)
            modules = new_modules
        return torch.nn.Sequential(modules)
