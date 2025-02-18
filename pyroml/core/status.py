from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pyroml.core.stage import Stage
    from pyroml.loop.base import Loop


class Status:
    def __init__(self, loop: "Loop"):
        self.loop = loop
        self.epoch = 1
        self.step = 1

    @property
    def stage(self) -> "Stage":
        return self.loop.stage

    def to_dict(self, json: bool = False) -> dict[str, Any]:
        d = dict(
            stage=self.loop.stage,
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
