from copy import deepcopy
from math import prod

import torch
from lightning_trainable import Trainable, TrainableHParams
from FrEIA.distributions import StandardNormalDistribution
from lightning_trainable.hparams import HParams
from lightning_trainable.trainable.trainable import auto_pin_memory
from torch.utils.data import DataLoader, IterableDataset

import fff.data
from fff.data.multivariate_student_t import MultivariateStudentT
from fff.model.en_graph_utils.position_feature_prior import PositionFeaturePrior
from fff.model.utils import TrainWallClock


class ModelHParams(HParams):
    data_dim: int
    cond_dim: int
    latent_dim: int


class BaseModelHParams(TrainableHParams):
    latent_distribution: dict = dict(
        name="normal"
    )
    data_set: dict
    noise: float | list = 0.0
    track_train_time: bool = False


class BaseModel(Trainable):
    """
    This class abstracts some basic functionalities of Free-form flows.

    TODO: Merge this with the FreeFormFlow class, there is no use in this abstraction.
    """
    hparams: BaseModelHParams

    def __init__(self, hparams: BaseModelHParams | dict):
        if not isinstance(hparams, BaseModelHParams):
            hparams = BaseModelHParams(**hparams)

        train_data, val_data, test_data = fff.data.load_dataset(**hparams.data_set)

        super().__init__(hparams, train_data=train_data, val_data=val_data, test_data=test_data)

        try:
            self._data_dim = train_data.data_dim
            self._data_cond_dim = train_data.cond_dim
        except AttributeError:
            data_sample = train_data[0]
            self._data_dim = prod(data_sample[0].shape)
            if len(data_sample) == 1:
                self._data_cond_dim = 0
            else:
                if len(data_sample[1].shape) != 1:
                    raise NotImplementedError("More than one condition dimension is not supported.")
                self._data_cond_dim = data_sample[1].shape[0]
        self.latents = {}

    def train_dataloader(self) -> DataLoader | list[DataLoader]:
        """
        Configures the Train DataLoader for Lightning. Uses the dataset you passed as train_data.

        @return: The DataLoader Object.
        """
        if self.train_data is None:
            return []
        kwargs = {}
        try:
            kwargs["collate_fn"] = self.train_data.collate_fn
        except AttributeError:
            pass
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=not isinstance(self.train_data, IterableDataset),
            pin_memory=auto_pin_memory(self.hparams.pin_memory, self.hparams.accelerator),
            num_workers=self.hparams.num_workers,
            **kwargs,
        )

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        """
        Configures the Validation DataLoader for Lightning. Uses the dataset you passed as val_data.

        @return: The DataLoader Object.
        """
        if self.val_data is None:
            return []
        kwargs = {}
        try:
            kwargs["collate_fn"] = self.train_data.collate_fn
        except AttributeError:
            pass
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=auto_pin_memory(self.hparams.pin_memory, self.hparams.accelerator),
            num_workers=self.hparams.num_workers,
            **kwargs,
        )

    def test_dataloader(self) -> DataLoader | list[DataLoader]:
        """
        Configures the Test DataLoader for Lightning. Uses the dataset you passed as test_data.

        @return: The DataLoader Object.
        """
        if self.test_data is None:
            return []
        kwargs = {}
        try:
            kwargs["collate_fn"] = self.train_data.collate_fn
        except AttributeError:
            pass
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=auto_pin_memory(self.hparams.pin_memory, self.hparams.accelerator),
            num_workers=self.hparams.num_workers,
            **kwargs,
        )

    def get_latent(self, device):
        if device not in self.latents:
            latent_hparams = deepcopy(self.hparams.latent_distribution)
            distribution_name = latent_hparams.pop("name")
            if distribution_name == "normal":
                self.latents[device] = StandardNormalDistribution(self.latent_dim, device=device)
            elif distribution_name == "student_t":
                df = self.hparams.latent_distribution["df"] * torch.ones(1, device=device)
                self.latents[device] = MultivariateStudentT(df, self.latent_dim)
            elif distribution_name == "position-feature-prior":
                from fff.data.qm9.models import DistributionNodes
                try:
                    nodes_dist = DistributionNodes(self.train_data.node_counts)
                except AttributeError:
                    # TODO
                    nodes_dist = DistributionNodes({
                        4: 1
                    })
                self.latents[device] = PositionFeaturePrior(**latent_hparams, nodes_dist=nodes_dist, device=device)
            else:
                raise ValueError(f"Unknown latent distribution: {self.hparams.latent_distribution['name']}")
        return self.latents[device]

    @property
    def latent_dim(self):
        return self.models[-1].hparams.latent_dim

    def is_conditional(self):
        return self._data_cond_dim != 0

    @property
    def cond_dim(self):
        soft_flow_cond_dim = 1 if isinstance(self.hparams.noise, list) else 0
        hp_aware_cond_dim = sum(
            1 if isinstance(weight, list) else 0
            for weight in self.hparams.loss_weights.values()
        )

        return self._data_cond_dim + soft_flow_cond_dim + hp_aware_cond_dim

    @property
    def data_dim(self):
        return self._data_dim

    def configure_callbacks(self):
        callbacks = super().configure_callbacks()
        if self.hparams.track_train_time:
            callbacks.append(TrainWallClock())
        return callbacks
