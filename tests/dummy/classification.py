import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torchmetrics import Metric
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall

from pyroml.model import PyroModel, Step
from pyroml.utils import Stage

from dummy.regression import DummyRegressionDataset


class DummyClassificationDataset(Dataset):
    def __init__(self, size=1024):
        self.ds = DummyRegressionDataset(size, 1)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        y = torch.where(y > 0, 1, 0).float()
        return x, y


class DummyClassificationModel(PyroModel):
    # Interestingly, with mid_dim=16 and seed=42, the model is naturally good at the dummy classification task :)
    def __init__(self, mid_dim=24):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(1, mid_dim),
            nn.LeakyReLU(),
            nn.Linear(mid_dim, 1),
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


if __name__ == "__main__":

    acc = 0
    size = 2048
    for x, y in DummyClassificationDataset(size=size):
        acc += y.sum().item()
    print(torch.allclose(acc / size, 0.5))
