import sys

sys.path.append("..")

import torch
import torch.nn as nn

import pyroml as p


from tests import WANDB_PROJECT
from dummy.classification import DummyClassificationDataset, DummyClassificationModel

if __name__ == "__main__":
    import os

    os.environ["WANDB_MODE"] = "offline"

    tr_ds = DummyClassificationDataset(size=1024)
    ev_ds = DummyClassificationDataset(size=64)
    model = DummyClassificationModel()

    trainer = p.Trainer(
        dtype=torch.bfloat16,
        loss=nn.BCEWithLogitsLoss(),
        lr=0.001,
        batch_size=16,
        max_epochs=8,
        evaluate=True,
        evaluate_every=4,
        wandb=True,
        wandb_project=WANDB_PROJECT,
        num_workers=0,
    )

    metrics = trainer.fit(model, tr_ds, ev_ds)
    print(metrics)
