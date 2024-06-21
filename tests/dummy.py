import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys

sys.path.append("..")

from pyroml.config import Config
from pyroml.trainer import Trainer
from pyroml.model import PyroModel
from pyroml.utils import Stage


class DummyClassification(PyroModel):
    def __init__(self, in_dim=16):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(in_dim * 2, in_dim),
            nn.LeakyReLU(),
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.seq(x)

    def step(self, batch, stage: Stage):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        return loss


class DummyRegressionModel(PyroModel):
    def __init__(self, in_dim=16):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(in_dim * 2, in_dim),
            nn.LeakyReLU(),
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.seq(x)

    def step(self, batch, stage: Stage):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        return loss


class DummyRegressionDataset(Dataset):
    def __init__(self, size=1024, in_dim=16):
        self.in_dim = in_dim
        self.data = torch.rand(size, in_dim)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        y = x**2 + 0.3 * x + 0.1
        return x, y


if __name__ == "__main__":
    ds = DummyRegressionDataset()
    model = DummyRegressionModel()
    loader = DataLoader(ds, batch_size=2, num_workers=0)
    for x, y in loader:
        print(x, y)
        print(x.shape, y.shape)
        output = model(x)
        print(output.shape)
        assert output.shape == y.shape
        break

    config = Config(
        name="dummy",
        max_iterations=256,
        lr=0.001,
        batch_size=16,
        num_workers=0,
        evaluate=False,
        wandb=False,
        verbose=True,
        compile=False,
    )
    trainer = Trainer(model, config)
    _, cp_path = trainer.fit(ds)
    new_trainer = Trainer.from_pretrained(trainer.model, config, cp_path, resume=False)
    new_trainer.fit(ds)
