from copy import deepcopy

from fff.loss import nll_surrogate
from fff.base import FreeFormBaseHParams, FreeFormBase, LogProbResult


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

    def surrogate_log_prob(self, x, c, **kwargs) -> LogProbResult:
        # Then compute JtJ
        config = deepcopy(self.hparams.log_det_estimator)
        estimator_name = config.pop("name")
        assert estimator_name == "surrogate"

        out = nll_surrogate(
            x,
            lambda _x: self.encode(_x, c),
            lambda z: self.decode(z, c),
            **kwargs
        )
        volume_change = out.surrogate

        latent_prob = self._latent_log_prob(out.z, c)
        return LogProbResult(
            out.z, out.x1, latent_prob + volume_change, out.regularizations
        )
