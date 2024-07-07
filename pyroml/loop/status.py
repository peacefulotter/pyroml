import pyroml as p
from pyroml.utils import Stage


class Status:
    def __init__(self, model: "p.PyroModel"):
        self.model: "p.PyroModel" = model
        self.epoch = 0
        self.step = 0

    def to_dict(self):
        return dict(
            epoch=self.epoch,
            step=self.step,
        )

    def advance_step(self, stage: Stage):
        self.step += 1

    def advance_epoch(self, stage: Stage):
        self.epoch += 1
