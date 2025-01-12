import os
import torch
import warnings
import torch.nn as nn
import safetensors.torch as st

from enum import Enum
from pathlib import Path
from typing import TypeAlias
from torchmetrics import Metric
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler

import pyroml as p
from pyroml.callback import Callback
from pyroml.checkpoint import Checkpoint
from pyroml.utils import Stage
from pyroml.hparams import WithHyperParameters

log = p.get_logger(__name__)


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
        super().__init__(hparams_file=Checkpoint.TRAINER_HPARAMS.value)
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
    ) -> dict[str, Metric] | None:
        pass

    def compile(self, *args, **kwargs):
        log.info(f"Compiling model...")
        model = torch.compile(self, *args, **kwargs)
        log.info(f"Model compiled!")
        return model

    def _setup(self, trainer: "p.Trainer"):
        self.trainer = trainer

    def configure_optimizers(self):
        """
        Define optimizer and optionally scheduler to use during training
        Trainer class can be accessed using self.trainer

        Example:
        ```
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.trainer.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)
        ```

        Default optimizer is SGD with learning rate from trainer.lr
        Default scheduler is None, meaning the learning rate will be constant
        """
        tr = self.trainer
        self.optimizer: Optimizer = torch.optim.SGD(self.parameters(), lr=tr.lr)

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
        if hasattr(self, "scheduler") and self.scheduler is not None:
            self.scheduler.step()

        # Zero the gradients
        self.optimizer.zero_grad(set_to_none=True)

        return loss

    def get_current_lr(self):
        if not hasattr(self, "scheduler") or self.scheduler is None:
            return self.trainer.lr
        return float(self.scheduler.get_last_lr()[0])

    def _is_compiled(self):
        return isinstance(self, torch._dynamo.OptimizedModule)

    def _get_module(self):
        """
        In case the model is compiled, use the _orig_mod attribute instead of the raw model
        """
        return getattr(self, "_orig_mod", self)

    def save(self, checkpoint_folder: Path | str, hparams_file=None):
        """
        Saves both model's weights and hyperparameters
        """
        self.save_checkpoint(folder=checkpoint_folder)
        self.save_hparams(folder=checkpoint_folder, file=hparams_file)

    def load(self, checkpoint_folder: Path | str, hparams_file=None):
        """
        Saves both model's weights and hyperparameters
        """
        self.load_checkpoint(folder=checkpoint_folder)
        self.load_hparams(folder=checkpoint_folder, file=hparams_file)

    def save_checkpoint(self, folder: Path | str):
        folder = Path(folder)
        os.makedirs(folder, exist_ok=True)

        f = folder / Checkpoint.MODEL_WEIGHTS.value
        log.info(f"Saving model weights to {f}")

        model = self._get_module()
        st.save_model(model=model, filename=f)

    def load_checkpoint(self, folder: Path | str = None, strict: bool = True):
        """
        Loads a pretrained model from a specified checkpoint folder.

        Args:
            folder (str): The folder path where the pretrained model is saved.
            strict (bool): Whether to strictly enforce that the model weights match the model architecture.
        """
        f = Path(folder) / Checkpoint.MODEL_WEIGHTS.value
        log.info(f"Loading model weights from {f}")

        model = self._get_module()
        missing, unexpected = st.load_model(model=model, filename=f, strict=strict)
        if not strict:
            warnings.warn(f"Missing layers: {missing}\nUnexpected layers: {unexpected}")
