import sys

sys.path.append("..")

import time
import random
import torch
from torch.utils.data import DataLoader

import pyroml as p
from pyroml import Loop
from pyroml.utils import Stage, seed_everything
from pyroml.metrics.tracker import MetricsTracker

from dummy import DummyClassificationModel, DummyClassificationDataset


if __name__ == "__main__":
    SEED = 42
    seed_everything(SEED)

    model = DummyClassificationModel()
    dataset = DummyClassificationDataset()
    loader = iter(DataLoader(dataset, batch_size=16, shuffle=True))

    trainer = p.Trainer()
    loop = p.TestLoop(trainer, model)
    tracker = MetricsTracker(loop)

    def step(stage: Stage, epoch: int, step: int):
        batch = next(loader)
        out = model.step(batch, stage)
        tracker.update(output=out)
        loop.status.advance_step()
        time.sleep(random.random() / 3)

    with torch.no_grad():
        for e in range(3):
            for s in range(10):
                step(Stage.TRAIN, e, s)
            for s in range(2):
                step(Stage.VAL, e, s)

            loop.status.advance_epoch()

    print(tracker.records)

    """
    from matplotlib import pyplot as plt
    tracker.plot(Stage.TRAIN)
    plt.show()
    tracker.plot(Stage.VAL)
    plt.show()"""
