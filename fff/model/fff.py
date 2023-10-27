from fff.model.base import FreeFormBaseHParams, FreeFormBase


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
