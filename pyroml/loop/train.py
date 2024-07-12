import logging
from torch.utils.data import Dataset

import pyroml as p
from pyroml.utils import Stage
from pyroml.loop.base import Loop
from pyroml.loop.eval import EvalLoop
from pyroml.wandb_logger import Wandb


log = logging.getLogger(__name__)


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
            log.warn(
                "You have chosen to evaluate the model, but no evaluation dataset is passed. Ignoring evaluation."
            )

        if trainer.wandb:
            wandb = Wandb(trainer=self.trainer, model=self.model, status=self.status)
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
            and self.status.step % self.trainer.evaluate_every == 0
        ):
            eval_loop = EvalLoop(trainer=self.trainer, model=self.model)
            eval_loop.run(self.ev_dataset)
            # TODO: save eval metrics in loop
            # eval_loop.tracker
            # remove eval progress bar, keep metrics

    def after_step(self, output: "p.StepOutput"):
        self.model._fit(output)

    def run(self, dataset: Dataset):
        self.model._configure_optimizers()

        self.model.to(self.autocast.device)
        super().run(dataset)
        self.model.cpu()
