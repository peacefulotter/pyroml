import os
import torch
import datasets
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torchmetrics.classification import Accuracy, Precision, Recall

from pyroml.utils import Stage
from pyroml.model import PyroModel, Step


class IrisNet(PyroModel):
    def __init__(self, mid_dim=16):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(4, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, 3),
        )

    def configure_metrics(self):
        return {
            "pre": Precision(task="multiclass", num_classes=3),
            "acc": Accuracy(task="multiclass", num_classes=3),
            "rec": Recall(task="multiclass", num_classes=3),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)

    def step(self, data: tuple[torch.Tensor], stage: Stage):
        x, y = data
        preds: torch.Tensor = self(x)

        metric_preds = torch.softmax(preds, dim=1)  # preds.argmax(dim=1)
        # metric_target = y.argmax(dim=1)

        return {
            Step.PRED: preds,
            Step.TARGET: y,
            Step.METRIC_PRED: metric_preds,
            # Step.METRIC_TARGET: metric_target,
        }


class IrisDataset(Dataset):
    def __init__(self, ds):
        super().__init__()
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


def load_dataset(folder="iris-data") -> datasets.arrow_dataset.Dataset:
    species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    def map_species(x):
        x["Species"] = species.index(x["Species"])
        return x

    if os.path.isdir(folder):
        ds = datasets.load_from_disk(folder)
    else:
        ds = datasets.load_dataset("scikit-learn/iris", split="train")
        ds.save_to_disk(folder)

    ds = ds.map(map_species)
    ds = ds.with_format("torch")
    return ds
