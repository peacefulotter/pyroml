import torch
import numpy as np

from torch.utils.data import Dataset


class DatasetWithNeighbors(Dataset):
    def __init__(
        self,
        dataset: "Dataset",
        neighbors: torch.Tensor,
        num_neighbors: int,
    ):
        super().__init__()
        self.dataset = dataset
        self.neighbors = neighbors
        self.num_neighbors = num_neighbors
        assert num_neighbors <= neighbors.shape[-1]

    def _get_neighbors(self, idx: int):
        neighbors_indices = np.random.choice(
            self.neighbors[idx], self.num_neighbors, replace=False
        )
        neighbors = torch.cat(
            [self.dataset[i]["img"].unsqueeze(0) for i in neighbors_indices],
            dim=0,
        )
        return neighbors

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item["neighbors"] = self._get_neighbors(idx)
        return item
