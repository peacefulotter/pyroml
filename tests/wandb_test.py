import sys

sys.path.append("..")

import torch
import torch.nn as nn

import pyroml as p


from dummy.classification import DummyClassificationDataset, DummyClassificationModel

if __name__ == "__main__":
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
        num_workers=0,
    )

    metrics = trainer.fit(model, tr_ds, ev_ds)
    print(metrics)
