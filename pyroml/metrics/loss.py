import torch
from torchmetrics import Metric

import pyroml as p


#     self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
#     self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

# def update(self, preds: Tensor, target: Tensor) -> None:
#     preds, target = self._input_format(preds, target)
#     if preds.shape != target.shape:
#         raise ValueError("preds and target must have the same shape")

#     self.correct += torch.sum(preds == target)
#     self.total += target.numel()

# def compute(self) -> Tensor:
#     return self.correct.float() / self.total


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
        avg_loss = 0
        if len(self.losses) > 0:
            avg_loss = sum(self.losses) / len(self.losses)
        return torch.tensor(avg_loss)

    def reset(self):
        self.losses = []
