try:
    from geomstats.geometry.manifold import Manifold
except ImportError:
    Manifold = object
from torch.utils.data import Dataset


class ManifoldDataset(Dataset):
    def __init__(self, dataset: Dataset, manifold: Manifold):
        self.dataset = dataset
        self.manifold = manifold

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
