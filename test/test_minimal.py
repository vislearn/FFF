import sys
print(sys.path)

import fff
import torch.nn


def test_minimal():
    encoder = torch.nn.Sequential(
        torch.nn.Linear(2, 32),
        torch.nn.SiLU(),
        torch.nn.Linear(32, 2),
    )
    decoder = torch.nn.Sequential(
        torch.nn.Linear(2, 32),
        torch.nn.SiLU(),
        torch.nn.Linear(32, 2),
    )

    latent_distribution = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(2), torch.ones(2)),
        1
    )

    loss = fff.loss.fff_loss(
        torch.randn(16, 2),
        encoder, decoder,
        latent_distribution,
        100, 1
    )
    loss.mean().backward()
