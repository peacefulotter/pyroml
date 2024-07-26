import torch
from torchmetrics import Metric

import pyroml as p


class LossMetric(Metric):
    def __init__(self, loop: "p.Loop"):
        super().__init__()
        self.loss_fn = loop.trainer.loss
        self.losses = []

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor, output: "p.StepOutput"):
        if "loss" in output:
            loss = output["loss"]
        else:
            loss = self.loss_fn(pred, target)

        self.losses.append(loss.item())
        return loss

    def compute(self):
        avg_loss = sum(self.losses) / len(self.losses)
        self.losses = []
        return torch.tensor(avg_loss)
