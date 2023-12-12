from torch.distributions import Distribution
from torch.nn import Module


class ModuleDistribution(Module):
    """
    A distribution that is also a module.

    This allows to use parameters of the distribution as parameters of the module.
    The parameters are stored in the module and the distribution is instantiated
    with these parameters. In order to fulfill constraints, you can derive sensible
    values from the parameters in instantiate().
    """
    def __init__(self):
        super().__init__()

    def instantiate(self) -> Distribution:
        raise NotImplementedError()

    def log_prob(self, x):
        return self.instantiate().log_prob(x)

    def rsample(self, sample_shape):
        return self.instantiate().rsample(sample_shape)

    def sample(self, sample_shape):
        return self.instantiate().sample(sample_shape)