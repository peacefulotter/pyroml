import warnings
from typing import TYPE_CHECKING, Optional

import torch
from torch.utils.data import Dataset
from typing_extensions import override

from pyroml.core.stage import Stage
from pyroml.loop.base import Loop
from pyroml.utils.log import get_logger

if TYPE_CHECKING:
    from pyroml.core.model import PyroModule
    from pyroml.core.trainer import Trainer

log = get_logger(__name__)


class TrainLoop(Loop):
    def __init__(
        self,
        trainer: "Trainer",
        model: "PyroModule",
        dataset: Dataset,
        ev_dataset: Optional[Dataset] = None,
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
        self.trainer._evaluate_from_train(
            model=self.model, dataset=self.ev_dataset, epoch=self.epoch
        )

    @override
    def on_train_iter_end(self, _):
        if (
            self.evaluate_enabled
            and self.trainer.evaluate_on == "step"
            and self.step % self.trainer.evaluate_every == 0
        ):
            self.evaluate()

    @override
    def on_train_epoch_end(self, _):
        if (
            self.evaluate_enabled
            and self.trainer.evaluate_on == "epoch"
            and self.epoch % self.trainer.evaluate_every == 0
        ):
            self.evaluate()

    @override
    def after_step(self, loss: torch.Tensor):
        self.model._fit(loss)

    @override
    def _run(self):
        # TODO: add way to evaluate before training and save eval progress bar
        # Probably want to add a sanitize loop
        # if self.evaluate_enabled:
        #    self.evaluate()

        self.model.configure_optimizers(self)
        self.model.train()
        res = super()._run()
        self.model.eval()
        return res
