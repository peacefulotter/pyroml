import sys

sys.path.append("..")

import torch
import numpy as np


from random import random

from pyroml.utils import Stage
from pyroml.tracker import Tracker


if __name__ == "__main__":
    tracker = Tracker()

    for epoch in range(3):
        for i in range(10):
            out = tracker.update(
                Stage.TRAIN, epoch, {"loss": torch.tensor([random()]), "acc": random()}
            )
            print(out)
        for i in range(2):
            out = tracker.update(
                Stage.VAL, epoch, {"loss": np.array([random()]), "f1": random()}
            )
            print(out)

    print(tracker.stats)
