from typing import Any

import pyroml as p


class Status:
    def __init__(self, loop: "p.Loop"):
        self.loop: "p.PyroModel" = loop
        self.epoch = 0
        self.step = 0

    @property
    def stage(self):
        return self.loop.stage

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
