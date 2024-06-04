from math import prod

import torch
from torch.nn import Flatten
from torch.nn.functional import one_hot
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10, CelebA
from torchvision.transforms import ToTensor, Resize, Compose, CenterCrop
from tqdm import tqdm

from fff.data.utils import TrainValTest

CELEBA_CACHE = {}


def get_mnist_datasets(root: str, digit: int = None, conditional: bool = False) -> TrainValTest:
    try:
        train_dataset = MNIST(root, train=True)
        test_dataset = MNIST(root, train=False)
    except RuntimeError:
        # Input with timeout
        if input("Download dataset? [y/n] ").lower() != "y":
            raise RuntimeError("Dataset not downloaded")
        train_dataset = MNIST(root, train=True, download=True)
        test_dataset = MNIST(root, train=False, download=True)

    return _process_img_data(train_dataset, None, test_dataset, label=digit, conditional=conditional)


def get_cifar10_datasets(root: str, label: int = None, conditional: bool = False) -> TrainValTest:
    try:
        train_dataset = CIFAR10(root, train=True)
        test_dataset = CIFAR10(root, train=False)
    except RuntimeError:
        # Input with timeout
        if input("Download dataset? [y/n] ").lower() != "y":
            raise RuntimeError("Dataset not downloaded")
        train_dataset = CIFAR10(root, train=True, download=True)
        test_dataset = CIFAR10(root, train=False, download=True)

    return _process_img_data(train_dataset, None, test_dataset, label=label, conditional=conditional)


def get_celeba_datasets(root: str, image_size: None | int = 64, load_to_memory: bool = False) -> TrainValTest:
    cache_key = (root, image_size, load_to_memory)
    if cache_key not in CELEBA_CACHE:
        if load_to_memory:
            train_dataset = celeba_to_memory(root, split='train', image_size=image_size)
            val_dataset = celeba_to_memory(root, split='valid', image_size=image_size)
            test_dataset = celeba_to_memory(root, split='test', image_size=image_size)

            train_dataset, val_dataset, test_dataset = _process_img_data(train_dataset, val_dataset, test_dataset)
        else:
            train_dataset = celeba_downloaded(root, split='train', image_size=image_size)
            val_dataset = celeba_downloaded(root, split='valid', image_size=image_size)
            test_dataset = celeba_downloaded(root, split='test', image_size=image_size)

        CELEBA_CACHE[cache_key] = train_dataset, val_dataset, test_dataset

    return CELEBA_CACHE[cache_key]


def celeba_downloaded(root: str, split: str, image_size: int | None):
    # This seems to be the standard procedure for CelebA
    # see https://github.com/tolstikhin/wae/blob/master/datahandler.py
    transforms = []
    if image_size is not None:
        transforms.append(CenterCrop(140))
        transforms.append(Resize(image_size))
    transforms.append(ToTensor())
    transforms.append(Flatten(0))
    transform = Compose(transforms)
    try:
        dataset = CelebA(root, split=split, transform=transform)
    except RuntimeError:
        # Input with timeout
        if input("Download dataset? [y/n] ").lower() != "y":
            raise RuntimeError("Dataset not downloaded")
        dataset = CelebA(root, split=split, download=True, transform=transform)
    return UnconditionalDataset(dataset)


class UnconditionalDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __getitem__(self, item):
        return self.dataset[item][0],

    def __len__(self):
        return len(self.dataset)


class MemoryCelebA:
    def __init__(self, data: torch.Tensor):
        self.data = data


def celeba_to_memory(root: str, split: str, image_size: None | int) -> MemoryCelebA:
    file_name = f"{root}/celeba_batches_{split}_{image_size}.pt"
    try:
        batches = torch.load(file_name)
    except FileNotFoundError:
        batches = []
        dataset = celeba_downloaded(root, split=split, image_size=image_size)
        for data, in tqdm(DataLoader(dataset, batch_size=128)):
            batches.append(data)
        torch.save(batches, file_name)
    return MemoryCelebA(torch.cat(batches, 0))


def _process_img_data(train_dataset, val_dataset, test_dataset, label=None, conditional: bool = False):
    # Data is (N, H, W, C)
    train_data = train_dataset.data
    if val_dataset is None:
        if len(train_data) > 40000:
            val_data_split = 10000
        else:
            val_data_split = len(train_data) // 6
        val_data = train_data[-val_data_split:]
        train_data = train_data[:-val_data_split]
    else:
        val_data = val_dataset.data
    test_data = test_dataset.data

    # To PyTorch tensors
    if not torch.is_tensor(train_data):
        train_data = torch.from_numpy(train_data)
        val_data = torch.from_numpy(val_data)
        test_data = torch.from_numpy(test_data)

    # Permute to (N, C, H, W)
    if train_data.shape[-1] in [1, 2, 3]:
        train_data = train_data.permute(0, 3, 1, 2)
        val_data = val_data.permute(0, 3, 1, 2)
        test_data = test_data.permute(0, 3, 1, 2)

    # Reshape to (N, D)
    data_size = prod(train_data.shape[1:])
    if train_data.shape != (train_data.shape[0], data_size):
        train_data = train_data.reshape(-1, data_size)
        val_data = val_data.reshape(-1, data_size)
        test_data = test_data.reshape(-1, data_size)

    # Normalize to [0, 1]
    if train_data.max() > 1:
        train_data = train_data / 255.
        val_data = val_data / 255.
        test_data = test_data / 255.

    # Labels
    if label is not None or conditional:
        train_targets = train_dataset.targets
        if val_dataset is None:
            train_targets = train_targets[:-val_data_split]
            val_targets = train_targets[-val_data_split:]
        else:
            val_targets = val_dataset.targets
        test_targets = test_dataset.targets

        if not torch.is_tensor(train_targets):
            train_targets = torch.tensor(train_targets)
            val_targets = torch.tensor(val_targets)
            test_targets = torch.tensor(test_targets)

        if label is not None:
            if conditional:
                raise ValueError("Cannot have conditional and label")
            train_data = train_data[train_targets == label]
            val_data = val_data[val_targets == label]
            test_data = test_data[test_targets == label]

    # Collect tensors for TensorDatasets
    train_data = [train_data]
    val_data = [val_data]
    test_data = [test_data]

    # Conditions
    if conditional:
        train_data.append(one_hot(train_targets, -1))
        val_data.append(one_hot(val_targets, -1))
        test_data.append(one_hot(test_targets, -1))

    return TensorDataset(
        *train_data
    ), TensorDataset(
        *val_data
    ), TensorDataset(
        *test_data
    )
