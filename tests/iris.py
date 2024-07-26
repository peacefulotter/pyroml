# Inspired by https://github.com/yangzhangalmo/pytorch-iris/blob/master/main.py

import os
import torch
import datasets
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torchmetrics.classification import Accuracy, Precision, Recall

import sys

sys.path.append("..")

from tests import WANDB_PROJECT

from pyroml.trainer import Trainer
from pyroml.utils import Stage, seed_everything
from pyroml.model import PyroModel, Step


class IrisNet(PyroModel):
    def __init__(self, mid_dim=16):
        super(IrisNet, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(4, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, 3),
            nn.Softmax(dim=1),
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

        metric_preds = preds.argmax(dim=1)
        metric_target = y.argmax(dim=1)

        return {
            Step.PRED: preds,
            Step.TARGET: y,
            Step.METRIC_PRED: metric_preds,
            Step.METRIC_TARGET: metric_target,
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
        y = (
            F.one_hot(
                self.y[idx],
                num_classes=3,
            )
            .squeeze()
            .float()
        )
        return x, y


if __name__ == "__main__":
    SEED = 42
    seed_everything(SEED)

    species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    def map_species(x):
        x["Species"] = species.index(x["Species"])
        return x

    IRIS_DATA_FOLDER = "iris-data"
    if os.path.isdir(IRIS_DATA_FOLDER):
        ds = datasets.load_from_disk(IRIS_DATA_FOLDER)
    else:
        ds = datasets.load_dataset("scikit-learn/iris", split="train")
        ds.save_to_disk(IRIS_DATA_FOLDER)

    ds = ds.map(map_species)
    ds = ds.with_format("torch")
    ds = ds.shuffle(seed=SEED)
    tr_ds, ev_ds, te_ds = np.split(ds, [int(0.6 * len(ds)), int(0.7 * len(ds))])

    tr_ds = IrisDataset(tr_ds)
    ev_ds = IrisDataset(ev_ds)
    te_ds = IrisDataset(te_ds)

    model = IrisNet()

    max_epochs = 16
    batch_size = 16

    trainer = Trainer(
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.AdamW,
        scheduler=torch.optim.lr_scheduler.OneCycleLR,
        scheduler_params={
            "max_lr": 0.005,
            "steps_per_epoch": (len(tr_ds) // batch_size) + 1,
            "epochs": max_epochs,
            "anneal_strategy": "cos",
            "cycle_momentum": False,
            "div_factor": 1e2,
            "final_div_factor": 0.05,
        },
        max_epochs=max_epochs,
        batch_size=batch_size,
        lr=0.002,
        evaluate=True,
        evaluate_every=2,
        wandb=True,
        wandb_project=WANDB_PROJECT,
        verbose=True,
        debug=True,
    )

    metrics = trainer.fit(model, tr_ds, ev_ds)
    print(metrics)

    metrics = trainer.test(te_ds)
    print(metrics)
