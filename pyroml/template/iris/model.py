import torch
import torch.nn as nn
from torchmetrics.classification import Accuracy, Precision, Recall

import pyroml as p
from pyroml.core.stage import Stage
from pyroml.metrics.accuracy import binary_accuracy


class IrisModel(p.PyroModel):
    def __init__(self, mid_dim=16):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(4, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, 3),
        )
        self.softmax = nn.Softmax()
        self.loss = nn.CrossEntropyLoss()

    def configure_metrics(self):
        return {
            "pre": Precision(task="multiclass", num_classes=3),
            "acc": Accuracy(task="multiclass", num_classes=3),
            "rec": Recall(task="multiclass", num_classes=3),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)

    def step(self, data: tuple[torch.Tensor], stage: p.Stage):
        x, y = data
        preds: torch.Tensor = self(x)

        # metric_preds = torch.softmax(preds, dim=1)  # preds.argmax(dim=1)
        # metric_target = y.argmax(dim=1)

        loss: torch.Tensor = self.loss(preds, y)

        preds = torch.softmax(preds, dim=1)
        preds = torch.argmax(preds, dim=1)
        self.log(dict(loss=loss.item(), acc=binary_accuracy(preds, y).item()))

        if stage != Stage.TRAIN:
            return preds

        return loss
