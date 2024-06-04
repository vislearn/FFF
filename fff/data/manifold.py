try:
    from geomstats.geometry.manifold import Manifold
except ImportError:
    Manifold = object
from torch.utils.data import Dataset, Subset


class ManifoldDataset(Dataset):
    def __init__(self, dataset: Dataset, manifold: Manifold):
        self.dataset = dataset
        self.manifold = manifold

    def __getattr__(self, item):
        try:
            return getattr(self.dataset, item)
        except AttributeError:
            if isinstance(self.dataset, Subset):
                return getattr(self.dataset.dataset, item)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
