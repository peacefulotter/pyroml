# Inspired by https://github.com/yangzhangalmo/pytorch-iris/blob/master/main.py


import rich
import torch
import torch.utils
import torch.utils.data

import pyroml as p
from pyroml.loop import Loop
from pyroml.template.iris import IrisDataset, IrisModel


class ScheduledIrisModel(IrisModel):
    def configure_optimizers(self, loop: "Loop"):
        tr = self.trainer
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=tr.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=tr.lr,
            total_steps=loop.total_steps,
            # steps_per_epoch=loop.steps_per_epochs,
            # epochs=tr.max_epochs,
            anneal_strategy="cos",
            cycle_momentum=False,
            div_factor=1e2,
            final_div_factor=0.05,
        )

    def step(self, data, stage):
        loss: torch.Tensor = super().step(data, stage)
        self.log(dict(loss=loss.item()))
        return loss


def test_iris_training():
    dataset = IrisDataset(split="train")
    tr_ds, te_ds = torch.utils.data.random_split(
        dataset, [int(len(dataset) * 0.7), int(len(dataset) * 0.3)]
    )

    model = ScheduledIrisModel()

    trainer = p.Trainer(
        device="cpu",
        compile=False,
        max_epochs=1,
        batch_size=16,
        lr=0.005,
        evaluate_on=False,
        wandb=False,
    )

    tr_tracker = trainer.fit(model, tr_ds)
    te_tracker = trainer.evaluate(model, te_ds)
    rich.print(tr_tracker.records)
    rich.print(te_tracker.records)
