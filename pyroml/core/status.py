from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyroml.core.stage import Stage


class Status:
    def __init__(self):
        self.epoch = 1
        self.step = 1

    @property
    def stage(self) -> "Stage":
        raise NotImplementedError

    def to_status_dict(self, json: bool = False) -> dict[str, Any]:
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
