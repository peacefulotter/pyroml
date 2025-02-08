import warnings
from typing import TYPE_CHECKING

import pandas as pd
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from pyroml.core.stage import Stage
from pyroml.loop.base import Loop
from pyroml.utils.log import get_logger

if TYPE_CHECKING:
    from pyroml.core.model import PyroModel
    from pyroml.core.trainer import Trainer

log = get_logger(__name__)


class TrainLoop(Loop):
    def __init__(
        self,
        trainer: "Trainer",
        model: "PyroModel",
        dataset: Dataset,
        ev_dataset: Dataset = None,
    ):
        super().__init__(trainer=trainer, model=model, dataset=dataset)
        self.ev_dataset = ev_dataset

        if self.trainer.eval_enabled and self.ev_dataset is None:
            warnings.warn(
                "You have chosen to evaluate the model, but no evaluation dataset is passed. Ignoring evaluation."
            )

        self.evaluate_enabled = (
            self.trainer.eval_enabled and self.ev_dataset is not None
        )

        # TODO: WandB already supports evaluation and prediction, why not include it for all loops
        if trainer.wandb:
            from pyroml.loggers.wandb_logger import Wandb

            wandb = Wandb(self)
            self.callbacks.append(wandb)

    @property
    def stage(self):
        return Stage.TRAIN

    @property
    def max_steps(self):
        return self.trainer.max_steps

    @property
    def max_epochs(self):
        return self.trainer.max_epochs

    @property
    def batch_size(self) -> int:
        return self.trainer.batch_size

    @property
    def num_workers(self) -> int:
        return self.trainer.num_workers

    def evaluate(self):
        ev_tracker = self.trainer.evaluate(model=self.model, dataset=self.ev_dataset)

        # Save recorded validation metrics to training records
        eval_records = ev_tracker.records
        eval_records["epoch"] = self.status.epoch
        self.tracker.records = pd.concat(
            objs=(self.tracker.records, eval_records), ignore_index=True
        )

        # Reset trainer loop to self
        self.trainer.loop = self

    @override
    def before_step(self):
        if (
            self.evaluate_enabled
            and self.trainer.evaluate_on == "step"
            and self.status.step % self.trainer.evaluate_every == 0
        ):
            self.evaluate()

    @override
    def after_epoch(self):
        if (
            self.evaluate_enabled
            and self.trainer.evaluate_on == "epoch"
            and self.status.epoch % self.trainer.evaluate_every == 0
        ):
            self.evaluate()

    @override
    def after_step(self, loss: torch.Tensor):
        self.model._fit(loss)

    @override
    def _run(self):
        # TODO: add way to evaluate before training and save eval progress bar
        # if self.evaluate_enabled:
        #    self.evaluate()

        self.model.configure_optimizers(self)
        self.model.train()

        return super()._run()
