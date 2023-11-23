from copy import deepcopy

from fff.loss import nll_surrogate
from fff.base import FreeFormBaseHParams, FreeFormBase, LogProbResult


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

        # Add additional nll terms if available
        for key, value in list(out.regularizations.items()):
            if key.startswith("vol_change_"):
                out.regularizations[key.replace("vol_change_", "nll_")] = -(latent_prob + value)

        return LogProbResult(
            out.z, out.x1, latent_prob + volume_change, out.regularizations
        )
