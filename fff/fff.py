from .base import FreeFormBaseHParams, FreeFormBase


class FreeFormFlowHParams(FreeFormBaseHParams):
    pass


class FreeFormFlow(FreeFormBase):
    """
    A FreeFormFlow is a normalizing flow consisting of a pair of free-form
    encoder and decoder.
    """
    hparams: FreeFormFlowHParams

    def __init__(self, hparams: FreeFormFlowHParams | dict):
        super().__init__(hparams)
        if self.data_dim != self.latent_dim:
            raise ValueError("Data and latent dimension must be equal for a FreeFormFlow.")

    def _make_latent(self, name, device, **kwargs):
        try:
            super()._make_latent(name, device, **kwargs)
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
