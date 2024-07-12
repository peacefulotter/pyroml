import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys

from torchmetrics import Metric
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall

sys.path.append("..")

from pyroml.model import PyroModel, Step
from pyroml.utils import Stage


class DummyClassificationModel(PyroModel):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def configure_metrics(self) -> dict[Metric]:
        return {
            "acc": BinaryAccuracy(),
            "pre": BinaryPrecision(),
            "rec": BinaryRecall(),
        }

    def forward(self, x):
        return self.seq(x)

    def step(self, batch, stage: Stage):
        x, y = batch
        pred = self(x)
        return {Step.PRED: pred, Step.METRIC_PRED: torch.round(pred), Step.TARGET: y}


class DummyRegressionModel(PyroModel):
    def __init__(self, in_dim=16):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(in_dim * 2, in_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.seq(x)

    def step(self, batch, stage: Stage):
        x, y = batch
        pred = self(x)
        return {Step.PRED: pred, Step.TARGET: y}


class DummyRegressionDataset(Dataset):
    def __init__(self, size=1024, in_dim=16):
        self.in_dim = in_dim
        self.data = torch.rand(size, in_dim)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        y = x**2 - 0.3 * x - 0.1
        return x, y


class DummyClassificationDataset(Dataset):
    def __init__(self, size=1024, in_dim=16):
        self.ds = DummyRegressionDataset(size, 1)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        y = torch.where(y > 0, 1, 0)
        return x, y


if __name__ == "__main__":

    for model, ds in [
        (DummyClassificationModel(), DummyClassificationDataset()),
        (DummyRegressionModel(), DummyRegressionDataset()),
    ]:

        loader = DataLoader(ds, batch_size=16, num_workers=0)

        x, y = next(iter(loader))
        print(x.shape, y.shape)
        output = model(x)
        print(output.shape)
        assert output.shape == y.shape
