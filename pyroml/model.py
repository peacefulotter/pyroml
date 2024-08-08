import torch
import logging
import torch.nn as nn
import safetensors.torch as st

from enum import Enum
from pathlib import Path
from torchmetrics import Metric
from typing import TypeAlias
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler

import pyroml as p
from pyroml.callback import Callback
from pyroml.checkpoint import Checkpoint
from pyroml.utils import Stage
from pyroml.hparams import WithHyperParameters

log = logging.getLogger(__name__)


class Step(Enum):
    PRED = "pred"
    TARGET = "target"
    METRIC_PRED = "metric_pred"
    METRIC_TARGET = "metric_target"


StepOutput: TypeAlias = dict[Step, torch.Tensor]


class MissingStepMethodException(Exception):
    pass


class MissingStepKeyException(Exception):
    pass


class PyroModel(WithHyperParameters, Callback, nn.Module):
    def __init__(self):
        super().__init__(hparams_file=Checkpoint.TRAINER_HPARAMS)
        self.trainer: "p.Trainer"
        self.optimizer: Optimizer
        self.scheduler: Scheduler | None
        self.device: torch.device = torch.device("cpu")

    def to(self, device: torch.device):
        self.device = device
        return super().to(device)

    def step(
        self,
        batch: torch.Tensor | dict[torch.Tensor] | tuple[torch.Tensor],
        stage: Stage,
    ) -> StepOutput:
        msg = "a step method must be implemented for your PyroModel model"
        raise MissingStepMethodException(msg)

    def configure_metrics(
        self,
    ) -> dict[Metric] | None:
        pass

    def _setup(self, trainer: "p.Trainer"):
        self.trainer = trainer

    def _configure_optimizers(self):
        tr = self.trainer

        self.optimizer: Optimizer = tr.optimizer(
            self.parameters(), lr=tr.lr, **tr.optimizer_params
        )

        self.scheduler: Scheduler | None = None
        if tr.scheduler is not None:
            self.scheduler = tr.scheduler(self.optimizer, **tr.scheduler_params)

    def _compute_loss(self, output: StepOutput) -> torch.Tensor:
        if Step.PRED not in output:
            raise MissingStepKeyException(
                f"Your model should return a Step.PRED tensor, corresponding to your model prediction, in the model' step method"
            )
        if Step.TARGET not in output:
            raise MissingStepKeyException(
                f"Your model should return a Step.TARGET tensor, corresponding to the dataset targets/labels, in the model' step method"
            )

        pred = output[Step.PRED]
        target = output[Step.TARGET]
        loss: torch.Tensor = self.trainer.loss(pred, target)
        return loss

    def _fit(self, output: StepOutput) -> torch.Tensor:
        """
        Perform a training step on the model using the output of the step method.
        Override this method if you wish to customize the training loop.

        1. Computes loss based on trainer.loss
        2. Computes gradient using loss.backward()
        3. Clip the gradients if necessary
        4. Step the optimizer, effectively backpropagating the loss
        5. Step the scheduler if available
        6. Zero the gradients
        """
        # Compute loss and gradient
        loss = self._compute_loss(output)
        loss.backward()

        # Clip the gradients
        if self.trainer.grad_norm_clip != None and self.trainer.grad_norm_clip != 0.0:
            nn.utils.clip_grad_norm_(self.parameters(), self.trainer.grad_norm_clip)

        # Backpropagate the loss
        self.optimizer.step()

        # Step the scheduler
        if self.scheduler:
            self.scheduler.step()

        # Zero the gradients
        self.optimizer.zero_grad(set_to_none=True)

        return loss

    def get_current_lr(self):
        if self.scheduler == None:
            return self.trainer.lr
        return float(self.scheduler.get_last_lr()[0])

    def _is_compiled(self):
        return isinstance(self, torch._dynamo.OptimizedModule)

    def save_checkpoint(self, checkpoint_folder, hparams_file=None):
        log.info(f"Saving model to {checkpoint_folder}")

        # Saving model weights
        st.save_model(self, checkpoint_folder / Checkpoint.MODEL_WEIGHTS)

        # Saving model hyperparameters
        self.save_hparams(folder=checkpoint_folder, file=hparams_file)

    def from_pretrained(self, folder: Path = None, strict: bool = True):
        """
        Loads a pretrained model from a specified checkpoint folder.

        Args:
            folder (str): The folder path where the pretrained model is saved.
            strict (bool): Whether to strictly enforce that the model weights match the model architecture.
        """
        log.info(f"Loading checkpoint from {folder}")

        # Load model weights
        # Must be done after creating the trainer if the model has been saved compiled
        # In that case, the model passed as parameter also needs to be compiled before
        missing, unexpected = st.load_model(
            model=self, filename=folder / Checkpoint.MODEL_WEIGHTS, strict=strict
        )
        if not strict:
            log.warn(f"Missing layers: {missing}\nUnexpected layers: {unexpected}")
