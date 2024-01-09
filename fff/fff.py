from math import prod

from .base import FreeFormBaseHParams, FreeFormBase, VolumeChangeResult


class FreeFormFlowHParams(FreeFormBaseHParams):
    pass


class FreeFormFlow(FreeFormBase):
    """
    A FreeFormFlow is a normalizing flow consisting of a pair of free-form
    encoder and decoder.
    """
    hparams: FreeFormFlowHParams

    def __init__(self, hparams: FreeFormFlowHParams | dict):
        if not isinstance(hparams, FreeFormFlowHParams):
            hparams = FreeFormFlowHParams(**hparams)
        super().__init__(hparams)
        if self.data_dim != self.latent_dim:
            raise ValueError("Data and latent dimension must be equal for a FreeFormFlow.")

    def _make_latent(self, name, device, **kwargs):
        try:
            return super()._make_latent(name, device, **kwargs)
        except ValueError:
            # Needed for QM9, only useful for dimension-preserving flows
            if name == "position-feature-prior":
                from fff.data.qm9.models import DistributionNodes
                try:
                    nodes_dist = DistributionNodes(self.train_data.node_counts)
                except AttributeError:
                    # TODO
                    nodes_dist = DistributionNodes({
                        4: 1
                    })
                from fff.model.en_graph_utils.position_feature_prior import PositionFeaturePrior
                return PositionFeaturePrior(**kwargs, nodes_dist=nodes_dist, device=device)
            else:
                raise

    def _encoder_volume_change(self, x, c, **kwargs) -> VolumeChangeResult:
        z, jac_enc = self._encoder_jac(x, c, **kwargs)
        jac_enc = jac_enc.reshape(x.shape[0], prod(z.shape[1:]), prod(x.shape[1:]))
        log_det = jac_enc.slogdet()[1]
        return VolumeChangeResult(z, log_det, {})

    def _decoder_volume_change(self, z, c, **kwargs) -> VolumeChangeResult:
        x1, jac_dec = self._decoder_jac(z, c, **kwargs)
        jac_dec = jac_dec.reshape(z.shape[0], prod(x1.shape[1:]), prod(z.shape[1:]))
        log_det = jac_dec.slogdet()[1]
        return VolumeChangeResult(x1, log_det, {})
