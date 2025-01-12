from typing import Any

import pyroml as p


class Status:
    def __init__(self, stage: "p.Stage"):
        self.stage = stage
        self.epoch = 0
        self.step = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(
            stage=self.stage.value,
            epoch=self.epoch,
            step=self.step,
        )

    def advance_step(self):
        self.step += 1

    def advance_epoch(self):
        self.epoch += 1

    def reset(self):
        self.epoch = 0
        self.step = 0
