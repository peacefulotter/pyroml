from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset, default_collate

from pyroml.core.stage import Stage
from pyroml.loop.eval import EvalLoop

if TYPE_CHECKING:
    from pyroml.core.model import PyroModel
    from pyroml.core.trainer import Trainer


class PredictLoop(EvalLoop):
    def __init__(self, trainer: "Trainer", model: "PyroModel", dataset: Dataset):
        super().__init__(trainer=trainer, model=model, dataset=dataset)
        self.predictions: list[Any] = []

    @property
    def max_steps(self):
        return None

    @property
    def stage(self):
        return Stage.PREDICT

    def after_step(self, output: torch.Tensor | Any):
        self.predictions.append(output)

    def _run(self):
        with torch.inference_mode():
            tracker = super()._run()

        preds = default_collate(self.predictions)
        return tracker, preds
