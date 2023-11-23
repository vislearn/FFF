from fff.base import FreeFormBaseHParams, FreeFormBase


class FreeFormInjectiveFlowHParams(FreeFormBaseHParams):
    pass


class FreeFormInjectiveFlow(FreeFormBase):
    """
    A FreeFormInjectiveFlow is an injective flow consisting of a pair of free-form
    encoder and decoder.
    """
    hparams: FreeFormInjectiveFlowHParams

    def __init__(self, hparams: FreeFormInjectiveFlowHParams | dict):
        super().__init__(hparams)
        if self.data_dim >= self.latent_dim:
            raise ValueError("Latent dimension must be less than data dimension "
                             "for a FreeFormInjectiveFlow.")
