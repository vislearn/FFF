from math import prod

import torch

from fff.base import FreeFormBaseHParams, FreeFormBase, VolumeChangeResult


class FreeFormInjectiveFlowHParams(FreeFormBaseHParams):
    pass


class FreeFormInjectiveFlow(FreeFormBase):
    """
    A FreeFormInjectiveFlow is an injective flow consisting of a pair of free-form
    encoder and decoder.
    """
    hparams: FreeFormInjectiveFlowHParams

    def __init__(self, hparams: FreeFormInjectiveFlowHParams | dict):
        if not isinstance(hparams, FreeFormInjectiveFlowHParams):
            hparams = FreeFormInjectiveFlowHParams(**hparams)
        super().__init__(hparams)
        if self.data_dim <= self.latent_dim:
            raise ValueError("Latent dimension must be less than data dimension "
                             "for a FreeFormInjectiveFlow.")

    def _encoder_volume_change(self, x, c, **kwargs) -> VolumeChangeResult:
        z, jac_enc = self._encoder_jac(x, c, **kwargs)
        jac_enc = jac_enc.reshape(x.shape[0], prod(z.shape[1:]), prod(x.shape[1:]))
        jtj = torch.einsum("bik,bjk->bij", jac_enc, jac_enc)
        log_det = jtj.slogdet()[1] / 2
        return VolumeChangeResult(z, log_det, {})

    def _decoder_volume_change(self, z, c, **kwargs) -> VolumeChangeResult:
        # Forward gradient is faster because latent dimension is smaller than data dimension
        x1, jac_dec = self._decoder_jac(z, c, grad_type="forward", **kwargs)
        jac_dec = jac_dec.reshape(z.shape[0], prod(x1.shape[1:]), prod(z.shape[1:]))
        jjt = torch.einsum("bki,bkj->bij", jac_dec, jac_dec)
        log_det = jjt.slogdet()[1] / 2
        return VolumeChangeResult(x1, log_det, {})
