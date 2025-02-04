import warnings
import pandas as pd
from torch.utils.data import Dataset

import pyroml as p
from pyroml.utils import Stage
from pyroml.loop.base import Loop
from pyroml.loop.eval import EvalLoop
from pyroml.wandb_logger import Wandb


log = p.get_logger(__name__)


class TrainLoop(Loop):

    def __init__(
        self,
        trainer: "p.Trainer",
        model: "p.PyroModel",
        ev_dataset: Dataset = None,
    ):
        super().__init__(trainer, model)
        self.ev_dataset = ev_dataset

        if self.trainer.evaluate and self.ev_dataset is None:
            warnings.warn(
                "You have chosen to evaluate the model, but no evaluation dataset is passed. Ignoring evaluation."
            )

        if trainer.wandb:
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

    def before_step(self):
        if (
            self.trainer.evaluate
            and self.ev_dataset is not None
            and self.status.step % self.trainer.evaluate_every == 0
        ):
            eval_loop = EvalLoop(trainer=self.trainer, model=self.model)
            eval_loop.run(self.ev_dataset)
            # Update training progress bar with eval metrics
            self.progress.update_metrics(
                self, eval_loop.progress.metrics, advance=0, stage=Stage.TRAIN
            )
            # Save recorded metrics
            eval_records = eval_loop.tracker.records
            eval_records["epoch"] = self.status.epoch
            self.tracker.records = pd.concat((self.tracker.records, eval_records))

    def after_step(self, output: "p.StepOutput"):
        loss = self.model._fit(output)
        # Register loss in output prevents the metrics tracker the need to compute it again
        output["loss"] = loss

    def on_train_start(self, **kwargs: "p.CallbackKwargs"):
        self.model.configure_optimizers(self)
        self.model.train()
