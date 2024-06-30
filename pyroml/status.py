import pyroml as p
from pyroml.utils import Stage


class _StageStatus:
    def __init__(self):
        self.epoch = 0
        self.step = 0


class Status:
    def __init__(self, trainer: "p.Trainer"):
        self.trainer = trainer
        self.model: "p.PyroModel" = trainer.model

        self._stages = {stage: _StageStatus() for stage in Stage}
        self._cur_stage: Stage = Stage.VAL

    @property
    def stage(self):
        return self._cur_stage

    @stage.setter
    def stage(self, stage: Stage):
        old_stage = self.stage
        self._cur_stage = stage

        if stage == Stage.TRAIN:
            self.model.train()
        else:
            self.model.eval()

        self.trainer._trigger_callback(
            "stage_change", stage_callback=False, old_stage=old_stage, new_stage=stage
        )

    @property
    def _stage_status(self):
        if self._cur_stage is None:
            raise ValueError("Current stage is not set")
        return self._stages[self._cur_stage]

    @property
    def epoch(self):
        return self._stage_status.epoch

    @property
    def step(self):
        return self._stage_status.step

    def to_dict(self):
        return dict(
            epoch=self._stage_status.epoch,
            step=self._stage_status.step,
        )

    def advance_step(self):
        self._stage_status.step += 1

    def advance_epoch(self):
        self._stage_status.epoch += 1

    def get_step(self):
        return self._stage_status.step

    def get_epoch(self):
        return self._stage_status.epoch
