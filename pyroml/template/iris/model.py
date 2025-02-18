import torch
import torch.nn as nn
from torchmetrics.classification import Accuracy, Precision, Recall

import pyroml as p
from pyroml.core.stage import Stage


class IrisModel(p.PyroModule):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.loss = nn.CrossEntropyLoss()

        self.pre = Precision(task="multiclass", num_classes=3)
        self.acc = Accuracy(task="multiclass", num_classes=3)
        self.rec = Recall(task="multiclass", num_classes=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)

    def step(self, data: tuple[torch.Tensor], stage: p.Stage):
        x, y = data
        preds: torch.Tensor = self(x)
        loss: torch.Tensor = self.loss(preds, y)

        preds = torch.softmax(preds, dim=-1)
        self.log(
            loss=loss.item(),
            acc=self.acc(preds, y),
            pre=self.pre(preds, y),
            rec=self.rec(preds, y),
        )

        if stage == Stage.TRAIN:
            return loss

        preds = torch.argmax(preds, dim=1)
        return preds
