import sys

sys.path.append("..")

import numpy as np
import torch
from torch.utils.data import DataLoader

import pyroml as p
from pyroml import Stage
from pyroml.core.tracker import MetricsTracker
from pyroml.loop import PredictLoop
from tests.dummy.classification import (
    DummyClassificationDataset,
    DummyClassificationModel,
)

if __name__ == "__main__":
    model = DummyClassificationModel()
    dataset = DummyClassificationDataset()
    loader = iter(DataLoader(dataset, batch_size=16, shuffle=True))

    trainer = p.Trainer()
    loop = PredictLoop(trainer, model)
    tracker = MetricsTracker(loop.status, model, trainer.loss)

    def step(stage: Stage, epoch: int, step: int):
        batch = next(loader)
        out = model.step(batch, stage)
        tracker.step(output=out)
        loop.status.advance_step()

    epochs = 3
    tr_steps_per_epoch = 10
    ev_steps_per_epoch = 2

    with torch.no_grad():
        for e in range(epochs):
            for s in range(tr_steps_per_epoch):
                step(Stage.TRAIN, e, s)
            for s in range(ev_steps_per_epoch):
                step(Stage.VAL, e, s)

            loop.status.advance_epoch()
            tracker.on_train_epoch_end(trainer, loop)

    assert hasattr(tracker, "records")
    rec = tracker.records
    print(rec)
    assert all(
        rec.columns
        == [
            "stage",
            "epoch",
            "step",
            "acc",
            "pre",
            "rec",
            "loss",
            "epoch_acc",
            "epoch_pre",
            "epoch_rec",
            "epoch_loss",
        ]
    )

    epoch_rec = tracker.get_epoch_records()
    assert epoch_rec.shape[0] == epochs

    step_rec = tracker.get_step_records()
    assert step_rec.shape[0] == epochs * (tr_steps_per_epoch + ev_steps_per_epoch)
    assert (
        len(step_rec["stage"].unique()) == 1
        and step_rec["stage"][0] == p.Stage.PREDICT.value
    )
    assert np.array_equal(np.arange(len(step_rec)), step_rec["step"])

    """
    from matplotlib import pyplot as plt
    tracker.plot(Stage.TRAIN)
    plt.show()
    tracker.plot(Stage.VAL)
    plt.show()"""
