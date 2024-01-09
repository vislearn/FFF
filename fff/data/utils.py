from typing import Tuple

import numpy as np

import torch.utils

TrainValTest = Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]

def split_dataset(data, seed=1241735):
    permuted = torch.from_numpy(np.random.default_rng(seed).permutation(data)).float()
    return (
        permuted[:int(0.8 * len(permuted))],
        permuted[int(0.8 * len(permuted)):int(0.9 * len(permuted))],
        permuted[int(0.9 * len(permuted)):]
    )