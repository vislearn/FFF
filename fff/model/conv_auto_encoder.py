from collections import OrderedDict
from math import prod

import torch
import torch.nn as nn

from fff.model.auto_encoder import SkipConnection
from fff.model.utils import guess_image_shape, wrap_batch_norm2d
from fff.base import ModelHParams


class ConvolutionalNeuralNetworkHParams(ModelHParams):
    skip_connection: bool = False
    ch_factor: int = 128

    encoder_spec: list = [
        [1, 4, 2, 1],
        [2, 4, 2, 1],
        [4, 4, 2, 1],
        [8, 4, 2, 1],
    ]
    # This decodes MNIST to 1x28x28 -- other decoders must be specified
    decoder_spec: list = [
        [8, 4],
        [4, 3, 2, 1],
        [2, 3, 2, 1, 1],
        [1, 3, 2, 1, 1],
    ]
    batch_norm: bool | str = False
    instance_norm: bool = False

    def __init__(self, **hparams):
        # Compatibility with old checkpoints
        if "encoder_fc_spec" in hparams:
            assert len(hparams["encoder_fc_spec"]) == 0
            del hparams["encoder_fc_spec"]
        if "decoder_fc_spec" in hparams:
            assert len(hparams["decoder_fc_spec"]) == 0
            del hparams["decoder_fc_spec"]
        if "decoder_spec" in hparams and len(hparams["decoder_spec"]) == 1:
            # This was the default for the first model
            hparams["decoder_spec"][0].append(4)

        super().__init__(**hparams)


class ConvolutionalNeuralNetwork(nn.Module):
    """
    This network contains two convolutional neural networks as encoder and decoder.
    """
    hparams: ConvolutionalNeuralNetworkHParams

    def __init__(self, hparams: dict | ConvolutionalNeuralNetworkHParams):
        if not isinstance(hparams, ConvolutionalNeuralNetworkHParams):
            hparams = ConvolutionalNeuralNetworkHParams(**hparams)

        super().__init__()
        self.hparams = hparams
        self.model = self.build_model()

    def cat_x_c(self, x, c):
        # Reshape as image, and concatenate conditioning as channel dimensions
        has_batch_dimension = len(x.shape) > 1
        if not has_batch_dimension:
            x = x[None, :]
            c = c[None, :]
        batch_size = x.shape[0]
        input_shape = guess_image_shape(self.hparams.data_dim)
        x_img = x.reshape(batch_size, *input_shape)
        c_img = c[:, :, None, None] * torch.ones(batch_size, self.hparams.cond_dim, *input_shape[1:], device=c.device)
        out = torch.cat([x_img, c_img], -3).reshape(batch_size, -1)
        if not has_batch_dimension:
            out = out[0]
        return out

    def encode(self, x, c):
        return self.model.encoder(self.cat_x_c(x, c))

    def decode(self, u, c):
        return self.model.decoder(torch.cat([
            u, c
        ], -1))

    def build_model(self):
        input_dim = self.hparams.data_dim
        input_shape = guess_image_shape(input_dim)
        cond_dim = self.hparams.cond_dim

        ch_factor = self.hparams.ch_factor
        encoder = nn.Sequential(
            nn.Unflatten(-1, (input_shape[0] + cond_dim, *input_shape[1:])),
        )
        tmp = encoder(self.cat_x_c(torch.randn(1, input_dim), torch.randn(1, cond_dim)))
        n_channels = input_shape[0] + cond_dim
        for i, conv_spec in enumerate(self.hparams.encoder_spec):
            out_channels, *args = conv_spec
            out_channels *= ch_factor
            conv = nn.Conv2d(n_channels, out_channels, *args)
            encoder.append(conv)
            if self.hparams.batch_norm is not False:
                encoder.append(wrap_batch_norm2d(self.hparams.batch_norm, out_channels))
            if self.hparams.instance_norm:
                encoder.append(nn.InstanceNorm2d(out_channels))
            n_channels = out_channels
            encoder.append(nn.ReLU())

            tmp = conv(tmp)
            if tmp.nelement() < self.hparams.latent_dim:
                raise ValueError(
                    f"Convolutional encoder layer {i} specified as {conv_spec} "
                    f"reduces data dimension to {tmp.nelement()}, which is "
                    f"less than the latent dimension {self.hparams.latent_dim}."
                )
        encoder.append(nn.Flatten(-3, -1))
        out_dim = tmp.nelement()

        encoder.append(nn.Linear(out_dim, self.hparams.latent_dim))

        # Start with ((ch_factor x channels_0 + cond_dim), latent_size, latent_size) image
        latent_channels, latent_size = self.hparams.decoder_spec[0]
        if isinstance(latent_size, int):
            latent_size = (latent_size, latent_size)
        latent_channels *= ch_factor
        n_channels = latent_channels + cond_dim

        decoder = nn.Sequential()
        decoder.append(nn.Linear(self.hparams.latent_dim + cond_dim, n_channels * prod(latent_size)))
        decoder.append(nn.Unflatten(-1, (n_channels, *latent_size)))
        tmp = decoder(torch.randn(1, self.hparams.latent_dim + cond_dim))
        for i, conv_spec in enumerate(self.hparams.decoder_spec):
            if i == 0:
                if len(conv_spec) != 2:
                    raise ValueError(f"First decoder layer must have only "
                                     f"an (out_channels, out_size) entry, but is: {conv_spec}")
            else:
                is_last_layer = i + 1 == len(self.hparams.decoder_spec)
                out_channels, *args = conv_spec
                if not is_last_layer:
                    out_channels *= ch_factor
                conv_transpose = nn.ConvTranspose2d(n_channels, out_channels, *args)
                decoder.append(conv_transpose)
                if not is_last_layer:
                    if self.hparams.batch_norm is not False:
                        decoder.append(wrap_batch_norm2d(self.hparams.batch_norm, out_channels))
                    if self.hparams.instance_norm:
                        decoder.append(nn.InstanceNorm2d(out_channels))
                decoder.append(
                    nn.ReLU()
                    if not is_last_layer else
                    nn.Sigmoid()
                )
                n_channels = out_channels

                tmp = conv_transpose(tmp)
                if tmp.nelement() < self.hparams.latent_dim:
                    raise ValueError(
                        f"Convolutional decoder layer {i} specified as {conv_spec} "
                        f"reduces data dimension to {tmp.nelement()}, which is "
                        f"less than the latent dimension {self.hparams.latent_dim}."
                    )

        # Check reconstruction dimensions
        out_shape = tmp.shape[1:]
        if out_shape != input_shape:
            raise ValueError(f"Decoder produces dimension {out_shape}, but "
                             f"{input_shape} was expected: {decoder}")
        decoder.append(nn.Flatten(-3, -1))

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
        return nn.Sequential(modules)
