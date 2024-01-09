# Free-form flows 

This is the official `PyTorch` implementation for our preprints:

1. [Free-form flows: Make Any Architecture a Normalizing Flow](http://arxiv.org/abs/2310.16624) on full-dimensional normalizing flows:
    ```bibtex
    @article{draxler2023freeform,
        title = {Free-form flows: Make Any Architecture a Normalizing Flow},
        author = {Draxler, Felix and Sorrenson, Peter and Rousselot, Armand and Zimmermann, Lea and Köthe, Ullrich},
        journal = {arXiv preprint arXiv:2310.16624},
        year = {2023}
    }
    ```
2. [Lifting Architectural Constraints of Injective Flows](http://arxiv.org/abs/2306.01843) on learning a manifold and the distribution on it jointly:
    ```bibtex
    @article{sorrenson2023lifting,
        title = {Lifting Architectural Constraints of Injective Flows},
        author = {Sorrenson, Peter and Draxler, Felix and Rousselot, Armand and Hummerich, Sander and Zimmermann, Lea and Köthe, Ullrich},
        journal = {arXiv preprint arXiv:2306.01843},
        year = {2023}
    }
    ```
3. Learning Distributions on Manifolds with Free-form Flows on learning distributions on a known manifold:
    ```bibtex
    @article{sorrenson2023learning,
        title = {Learning Distributions on Manifolds with Free-form Flows},
        author = {Sorrenson, Peter and Draxler, Felix and Rousselot, Armand and Hummerich, Sander and Köthe, Ullrich},
        journal = {on your favorite preprint server soon},
        year = {2023}
    }
    ```


## Installation

Options:

1. Install via `pip` and use our package.
2. Copy the loss script with `PyTorch` as the only dependency to train in your own setup.

### Install via pip

The following will install our package along with all of its dependencies:

```bash
git clone https://github.com/vislearn/FFF.git
cd FFF
pip install -r requirements.txt
pip install .
```

In the last line, use `pip install -e .` if you want to edit the code.

Then you can import the package via

```python
import fff
```

### Copy `fff/loss.py` into your project

If you do not want to add our `fff` package as a dependency,
but still want to use the FFF or FIF loss function,
you can copy the `fff/loss.py` file into your own project.
It does not have any dependencies except for PyTorch.


## Basic usage

### Train your architecture 

```python
import torch
import fff.loss as loss


class FreeFormFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(...)
        self.decoder = torch.nn.Sequential(...)


model = FreeFormFlow()
optim = ...
data_loader = ...
n_epochs = ...
beta = ...

for epoch in range(n_epochs):
    for batch in data_loader:
        optim.zero_grad()
        loss = loss.fff_loss(batch, model.encoder, model.decoder, beta)
        loss.backward()
        optim.step()
```


### Reproduce our experiments

All training configurations from our papers can be found in the `configs/(fff|fif)` directories.

Our training framework is built on [lightning-trainable](https://github.com/LarsKue/lightning-trainable), a configuration wrapper around [PyTorch Lightning](https://lightning.ai/pytorch-lightning). There is no `main.py`, but you can train all our models via the `lightning_trainable.launcher.fit` module.
For example, to train the Boltzmann generator on DW4:
```bash
python -m lightning_trainable.launcher.fit configs/fff/dw4.yaml --name '{data_set[name]}'
```

This will create a new directory `lightning_logs/dw4/`. You can monitor the run via `tensorboard`:
```bash
tensorboard --logdir lightning_logs
```

When training has finished, you can import the model via
```python
import fff

model = fff.FreeFormFlow.load_from_checkpoint(
    'lightning_logs/dw4/version_0/checkpoints/last.ckpt'
)
```

If you want to overwrite the default parameters, you can add `key=value`-pairs after the config file:
```bash
python -m lightning_trainable.launcher.fit configs/fff/dw4.yaml batch_size=128 loss_weights.noisy_reconstruction=20 --name '{data_set[name]}'
```

#### Known issues

Training with $E(n)$-GNNs is sometimes unstable. This is usually caught with an assertion in a later step and training is stopped.
In almost all cases, training can be stably resumed from the last epoch checkpoint by passing the `--continue-from [CHECKPOINT]` flag to the training, such as:
```bash
python -m lightning_trainable.launcher.fit configs/fff/dw4.yaml --name '{data_set[name]}' --continue-from lightning_logs/dw4/version_0/checkpoints/last.ckpt
```
This reloads the entire training state (model state, optim state, epoch, etc.) from the checkpoint and continues training from there.


### Setup your own training

Start with the config file in `configs/(fff|fif)` that fits your needs best and modify it.
For custom data sets, add the data set to `fff.data`.
