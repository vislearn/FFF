import torch
from torch.distributions import Distribution


class MixtureDistribution(Distribution):
    arg_constraints = {}

    def __init__(
        self,
        mixture_distribution,
        component_distributions,
        *args,
        finite_support: bool = False,
        **kwargs
    ):
        self.mixture_distribution = mixture_distribution
        self.component_distributions = component_distributions
        self.finite_support = finite_support

        super().__init__(*args, **kwargs)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        mixture_samples = self.mixture_distribution.sample(sample_shape)
        samples = torch.stack(
            [distr.sample(sample_shape) for distr in self.component_distributions]
        )
        samples = samples[mixture_samples, torch.arange(samples.shape[1])]
        return samples

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self.finite_support:

            probs = torch.zeros_like(value)
            for i, distr in enumerate(self.component_distributions):
                in_supp = distr.support.check(value).all(dim=-1)

                in_supp_probs = (
                    distr.log_prob(value[in_supp, ...]).exp()
                    * self.mixture_distribution.probs[i]
                )

                if in_supp_probs.dim() == 1:
                    in_supp_probs = in_supp_probs.unsqueeze(-1).expand_as(
                        value[in_supp, ...]
                    )

                probs[in_supp] += in_supp_probs

            log_probs = torch.log(probs)

        else:
            log_probs = torch.stack(
                [distr.log_prob(value) for distr in self.component_distributions]
            )
            log_probs = torch.logsumexp(log_probs, dim=0)

        return log_probs
