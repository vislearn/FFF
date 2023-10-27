from typing import Tuple

import torch.utils

TrainValTest = Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]
