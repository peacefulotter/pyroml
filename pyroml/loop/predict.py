from typing import TYPE_CHECKING, Any, Callable, Dict, Tuple, Type, Union

import torch
import torch.utils.data._utils.collate
from torch.utils.data import Dataset

from pyroml.core.stage import Stage
from pyroml.loop.eval import EvalLoop

if TYPE_CHECKING:
    from pyroml.core.model import PyroModel
    from pyroml.core.trainer import Trainer


def custom_collate_tensor_fn(batch, *, collate_fn_map):
    try:
        return torch.stack(batch, 0)
    except Exception:
        return torch.cat(batch, 0)


custom_collate_fn_map: Dict[Union[Type, Tuple[Type, ...]], Callable] = {
    torch.Tensor: custom_collate_tensor_fn
}


def custom_collate(batch):
    return torch.utils.data._utils.collate.collate(
        batch, collate_fn_map=custom_collate_fn_map
    )


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

        preds = custom_collate(self.predictions)
        return tracker, preds
