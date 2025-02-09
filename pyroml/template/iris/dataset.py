from pathlib import Path

import torch

from pyroml.template.base import TemplateDataset

IRIS_SPECIES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


def map_species(x):
    x["Species"] = IRIS_SPECIES.index(x["Species"])
    return x


class IrisDataset(TemplateDataset):
    def __init__(
        self, folder: Path | str = "data/iris", split: str = "train", save: bool = True
    ):
        super().__init__(
            dataset_path="scikit-learn/iris",
            folder=folder,
            transform=None,
            split=split,
            save=save,
        )

        # TODO: don't save x and y but compute them in __getitem__
        self.x = torch.empty(len(self.ds), 4)
        self.y = torch.zeros(len(self.ds), 1, dtype=torch.int64)
        for i, iris in enumerate(self.ds):
            self.x[i] = torch.stack(
                (
                    iris["SepalLengthCm"],
                    iris["SepalWidthCm"],
                    iris["PetalLengthCm"],
                    iris["PetalWidthCm"],
                )
            )
            self.y[i] = iris["Species"]

    def load_dataset(self):
        ds = self._load_dataset()
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
