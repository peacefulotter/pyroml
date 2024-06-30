import sys

sys.path.append("..")

import torch
from torch.utils.data import DataLoader

from pyroml.utils import Stage, seed_everything
from pyroml.metrics_tracker import MetricsTracker

from dummy import DummyClassificationModel, DummyClassificationDataset


if __name__ == "__main__":
    SEED = 42
    seed_everything(SEED)

    model = DummyClassificationModel()
    dataset = DummyClassificationDataset()
    loader = iter(DataLoader(dataset, batch_size=16, shuffle=True))
    tracker = MetricsTracker(model)

    def step(stage: Stage, epoch: int, step: int):
        batch = next(loader)
        out = model.step(batch, stage)
        tracker.update(stage=stage, output=out, epoch=epoch, step=step)

    with torch.no_grad():
        for e in range(3):
            for s in range(10):
                step(Stage.TRAIN, e, s)
            for s in range(2):
                step(Stage.VAL, e, s)

    print(tracker.records[Stage.TRAIN])
    print(tracker.records[Stage.VAL])

    print(tracker.metrics[Stage.TRAIN]["acc"])

    """
    from matplotlib import pyplot as plt
    tracker.plot(Stage.TRAIN)
    plt.show()
    tracker.plot(Stage.VAL)
    plt.show()"""
