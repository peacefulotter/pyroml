import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import safetensors.torch as st
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler

from pyroml.callbacks import Callback
from pyroml.core.hparams import WithHyperParameters
from pyroml.core.stage import Stage
from pyroml.utils.log import get_logger

if TYPE_CHECKING:
    from pyroml.core.trainer import Trainer
    from pyroml.loop.base import Loop


log = get_logger(__name__)


MODEL_WEIGHTS_FILE = "weights.safetensors"
MODEL_HPARAMS_FILE = "model_hparams.json"


class MissingStepMethodException(Exception):
    pass


class PyroModel(WithHyperParameters, Callback, nn.Module):
    def __init__(self):
        super().__init__(hparams_file=MODEL_HPARAMS_FILE)
        self.trainer: "Trainer"
        self.optimizer: Optimizer
        self.scheduler: Scheduler | None

    @property
    def device(self):
        return next(self.parameters()).device

    def configure_optimizers(self, loop: "Loop"):
        """
        Define optimizer and optionally scheduler to use during training

        Default values:
        - Optimizer is SGD with learning rate from trainer.lr
        - Scheduler is None, meaning the learning rate will be constant

        Note:
        Trainer class can be accessed using self.trainer
        If you have multiple optimizers / schedulers, store them in variables and override _fit

        Example:
        ```
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.trainer.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)
        ```
        """
        tr = self.trainer
        self.optimizer: Optimizer = torch.optim.SGD(self.parameters(), lr=tr.lr)

    def step(
        self,
        batch: Any,
        stage: Stage,
    ) -> torch.Tensor:
        msg = "A step method must be implemented for your PyroModel model"
        raise MissingStepMethodException(msg)

    def log(self, **data: dict[str, float | np.ndarray | torch.Tensor]):
        return self.trainer.log(**data)

    def compile(self, *args, **kwargs):
        log.info("Compiling model...")
        model = torch.compile(self, *args, **kwargs)
        log.info("Model compiled!")
        return model

    def _setup(self, trainer: "Trainer"):
        self.trainer = trainer

    def _fit(self, loss: torch.Tensor):
        """
        Perform a training step on the model using the output of the step method.
        => Override this method if you wish to customize the training loop; especially if you have multiple optimizers / schedulers

        By default, this method does the following:
        1. Backpropagate loss to model
        2. Clip the gradients if necessary
        3. Step the optimizer
        4. Step the scheduler if available
        """
        # TODO: add gradscaler: torch.amp.GradScaler(enabled=self.trainer.dtype == torch.float16) # bfloat16 too ?

        # 1. Backpropagate the loss
        loss.backward()

        # 2. Clip the gradients
        if (
            self.trainer.grad_norm_clip is not None
            and self.trainer.grad_norm_clip > 0.0
        ):
            nn.utils.clip_grad_norm_(self.parameters(), self.trainer.grad_norm_clip)

        # 3. Optimizer step
        self.optimizer.step()

        # 4. Step the scheduler
        if hasattr(self, "scheduler") and self.scheduler is not None:
            self.scheduler.step()

        self.optimizer.zero_grad(set_to_none=True)

    def get_current_lr(self) -> dict[str, float]:
        """
        In the event where you have multiple schedulers, override this method

        Returns:
            dict[str, float]: mapping of learning rate names to their corresponding values
        """
        if not hasattr(self, "scheduler") or self.scheduler is None:
            lr = self.trainer.lr
        else:
            lr = float(self.scheduler.get_last_lr()[0])
        return dict(lr=lr)

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
        self.save_weights(folder=checkpoint_folder)
        self.save_hparams(folder=checkpoint_folder, file=hparams_file)

    def load(self, checkpoint_folder: Path | str, hparams_file=None):
        """
        Loads both model's weights and hyperparameters
        """
        self.load_weights(folder=checkpoint_folder)
        self.load_hparams(folder=checkpoint_folder, file=hparams_file)

    def save_weights(self, folder: Path | str, file: Path | str = MODEL_WEIGHTS_FILE):
        """
        Saves model weights from a specified checkpoint folder.

        Args:
            folder (str): The folder path where the pretrained model is saved.
            file (str): The filename of the file containing the model weights
        """
        folder = Path(folder)
        os.makedirs(folder, exist_ok=True)

        f = folder / file
        log.info(f"Saving model weights to {f}")

        model = self._get_module()
        st.save_model(model=model, filename=f)

    def load_weights(
        self,
        folder: Path | str = None,
        file: Path | str = MODEL_WEIGHTS_FILE,
        strict: bool = True,
    ):
        """
        Loads model weights from a specified checkpoint folder.

        Args:
            folder (str): The folder path where the pretrained model is saved.
            file (str): The filename of the file containing the model weights
            strict (bool): Whether to strictly enforce that the model weights match the model architecture.
        """
        f = Path(folder) / file
        log.info(f"Loading model weights from {f}")

        model = self._get_module()
        missing, unexpected = st.load_model(model=model, filename=f, strict=strict)
        if not strict:
            warnings.warn(f"Missing layers: {missing}\nUnexpected layers: {unexpected}")
