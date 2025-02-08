from typing import Any

import pyroml as p


class Status:
    def __init__(self, stage: "p.Stage"):
        self.stage = stage
        self.reset()

    def reset(self):
        self.epoch = 1
        self.step = 1

    def to_dict(self, json: bool = False) -> dict[str, Any]:
        d = dict(
            stage=self.stage,
            epoch=self.epoch,
            step=self.step,
        )
        if json:
            d["stage"] = d["stage"].value
        return d

    def advance_step(self):
        self.step += 1

    def advance_epoch(self):
        self.epoch += 1
