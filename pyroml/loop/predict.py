from typing import TYPE_CHECKING, Any, Callable, Dict, Tuple, Type, Union
import warnings

import torch
import torch.utils.data._utils.collate
from torch.utils.data import Dataset, DataLoader

from pyroml.core.stage import Stage
from pyroml.loop.eval import EvalLoop
from pyroml.utils.device import to_device

if TYPE_CHECKING:
    from pyroml.core.model import PyroModule
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
    def __init__(self, trainer: "Trainer", model: "PyroModule", dataset: Dataset):
        super().__init__(trainer=trainer, model=model, dataset=dataset)
        self.predictions: torch.Tensor | list[Any] = []
        self.pred_idx = 0

    def on_predict_start(self, _):
        """
        Before running the prediciton loop, we prepare the predictions tensor
        by constructing an empty tensor with the proper size
        """
        loader = DataLoader(self.dataset, batch_size=1)
        batch = next(iter(loader))
        if self.trainer.auto_move:
            batch = to_device(batch, device=self.trainer.device)

        with torch.no_grad():
            preds = self.model.step(batch, stage=self.stage)

        if isinstance(preds, torch.Tensor):
            size = (len(self.dataset), *preds.shape[1:])
            self.predictions = torch.empty(size, device=self.trainer.device)
        else:
            warnings.warn(
                "Your model step method did not return a tensor during prediction, we thus cannot predict the final predictions shape which might be much slower"
            )
            self.predictions = []

    @property
    def max_steps(self):
        return None

    @property
    def stage(self):
        return Stage.PREDICT

    def after_step(self, output: torch.Tensor):
        if isinstance(self.predictions, list):
            self.predictions.append(output)
        else:
            batch_size = output.shape[0]
            i = self.pred_idx
            self.predictions[i : i + batch_size] = output
            self.pred_idx += batch_size

    def _run(self):
        tracker = super()._run()
        preds = (
            custom_collate(self.predictions)
            if isinstance(self.predictions, list)
            else self.predictions
        )
        return tracker, preds
