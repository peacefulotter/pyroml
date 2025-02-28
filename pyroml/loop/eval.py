from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from pyroml.core.stage import Stage
from pyroml.loop.base import Loop

if TYPE_CHECKING:
    from pyroml.core.model import PyroModule
    from pyroml.core.trainer import Trainer


class EvalLoop(Loop):
    def __init__(
        self, trainer: "Trainer", model: "PyroModule", dataset: Dataset, epoch: int = 1
    ):
        super().__init__(trainer=trainer, model=model, dataset=dataset)
        self.epoch = epoch

    @property
    def stage(self):
        return Stage.VAL

    @property
    def max_steps(self):
        return self.trainer.eval_max_steps

    @property
    def max_epochs(self):
        return 1

    @property
    def batch_size(self) -> int:
        return self.trainer.eval_batch_size

    @property
    def num_workers(self) -> int:
        return self.trainer.eval_num_workers

    def _run(self):
        self.model.eval()
        with torch.no_grad():
            return super()._run()
