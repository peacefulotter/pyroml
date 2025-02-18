import random
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchmetrics import Metric
from torchmetrics.regression import MeanSquaredLogError

import pyroml as p
from pyroml import Stage
from pyroml.core.model import PyroModule


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


class DummyRegressionModel(PyroModule):
    def __init__(self, in_dim=16, sleeping=False):
        super().__init__()
        self.in_dim = in_dim
        self.sleeping = sleeping
        self.seq = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(in_dim * 2, in_dim),
            nn.LeakyReLU(),
        )
        self.loss = nn.MSELoss()

    def configure_optimizers(self, loop: "p.Loop"):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.trainer.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.trainer.lr,
            total_steps=loop.total_steps,
            steps_per_epoch=loop.steps_per_epochs,
            epochs=self.trainer.max_epochs,
            anneal_strategy="cos",
            cycle_momentum=False,
            div_factor=1e2,
            final_div_factor=0.05,
        )

    def configure_metrics(self) -> dict[Metric]:
        return {
            "msle": MeanSquaredLogError(),
        }

    def forward(self, x):
        if self.sleeping:
            time.sleep(random.random())
        return self.seq(x)

    def step(self, batch, stage: Stage):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        return loss
