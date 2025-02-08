from pathlib import Path

import torch
from torch.utils.data import Dataset

from pyroml.template.base import load_dataset

IRIS_SPECIES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


class IrisDataset(Dataset):
    def __init__(
        self, folder: Path | str = "data/iris", split: str = "train", save: bool = True
    ):
        super().__init__()
        self.folder = folder
        self.split = split
        self.save = save

        ds = self._load_dataset()

        self.x = torch.empty(len(ds), 4)
        self.y = torch.zeros(len(ds), 1, dtype=torch.int64)
        for i, iris in enumerate(ds):
            self.x[i] = torch.stack(
                (
                    iris["SepalLengthCm"],
                    iris["SepalWidthCm"],
                    iris["PetalLengthCm"],
                    iris["PetalWidthCm"],
                )
            )
            self.y[i] = iris["Species"]

    def _load_dataset(self):
        def map_species(x):
            x["Species"] = IRIS_SPECIES.index(x["Species"])
            return x

        ds = load_dataset(
            dataset="scikit-learn/iris",
            folder=self.folder,
            split=self.split,
            save=self.save,
        )
        ds = ds.map(map_species)
        return ds

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx].float()
        y = self.y[idx].squeeze().long()
        # y = (
        #     F.one_hot(
        #         self.y[idx],
        #         num_classes=3,
        #     )
        #     .squeeze()
        #     .float()
        # )
        return x, y
