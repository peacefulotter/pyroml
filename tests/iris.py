# Inspired by https://github.com/yangzhangalmo/pytorch-iris/blob/master/main.py


import rich
import torch
import logging
import numpy as np
import torch.nn as nn


import sys

import torch.utils
import torch.utils.data

sys.path.append("..")

from setup import setup_test, DEFAULT_SEED

import pyroml as p
from pyroml.template.iris import IrisNet, IrisDataset, load_dataset
from pyroml.trainer import Trainer


class ScheduledIrisNet(IrisNet):
    def configure_optimizers(self, loop: "p.Loop"):
        tr = self.trainer
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=tr.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=tr.lr,
            total_steps=loop.total_steps,
            steps_per_epoch=loop.steps_per_epochs,
            epochs=tr.max_epochs,
            anneal_strategy="cos",
            cycle_momentum=False,
            div_factor=1e2,
            final_div_factor=0.05,
        )


class TestCallback(p.Callback):
    def __init__(self, te_ds: torch.utils.data.Dataset):
        self.te_ds = te_ds

    def on_train_epoch_end(
        self, trainer: Trainer, loop: p.Loop, **kwargs: p.CallbackKwargs
    ):
        metrics = trainer.test(trainer.model, self.te_ds)
        metrics.to_csv(f"iris_epoch={loop.status.epoch}.csv", index=False)


if __name__ == "__main__":
    setup_test()

    ds = load_dataset()
    ds = ds.shuffle(seed=DEFAULT_SEED)
    tr_ds, ev_ds, te_ds = np.split(ds, [int(0.6 * len(ds)), int(0.7 * len(ds))])

    tr_ds = IrisDataset(tr_ds)
    ev_ds = IrisDataset(ev_ds)
    te_ds = IrisDataset(te_ds)

    model = ScheduledIrisNet()

    trainer = Trainer(
        compile=True,
        loss=nn.CrossEntropyLoss(),
        max_epochs=16,
        batch_size=16,
        lr=0.005,
        evaluate=True,
        evaluate_every=12,
        wandb=False,
        log_level=logging.INFO,
        callbacks=[TestCallback(te_ds)],
    )

    tr_metrics = trainer.fit(model, tr_ds, ev_ds)
    te_metrics = trainer.test(model, te_ds)
    rich.print(tr_metrics)
    rich.print(te_metrics)
