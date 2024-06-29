import sys

sys.path.append("..")

import torch
import torch.nn as nn

import pyroml as p

from tests import WANDB_PROJECT
from dummy import DummyClassificationModel, DummyClassificationDataset

if __name__ == "__main__":
    tr_ds = DummyClassificationDataset(size=64)
    ev_ds = DummyClassificationDataset(size=16)
    model = DummyClassificationModel()

    config = p.Config(
        dtype=torch.bfloat16,
        loss=nn.CrossEntropyLoss(),
        batch_size=4,
        max_epochs=8,
        evaluate=True,
        evaluate_every=2,
        wandb=True,
        wandb_project=WANDB_PROJECT,
        num_workers=0,
    )
    trainer = p.Trainer(model, config)
    trainer.fit(tr_ds, ev_ds)
    print(trainer.tracker.records[p.Stage.TRAIN])
