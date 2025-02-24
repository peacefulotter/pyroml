import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import safetensors.torch as st
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler

from pyroml.callbacks import Callback
from pyroml.core.hparams import WithHyperParameters
from pyroml.core.stage import Stage
from pyroml.utils.classes import get_classname
from pyroml.utils.log import get_logger

if TYPE_CHECKING:
    from pyroml.core.trainer import Trainer
    from pyroml.loop.base import Loop


log = get_logger(__name__)


MODEL_WEIGHTS_FILE = "model_weights.safetensors"
MODEL_HPARAMS_FILE = "model_hparams.json"
MODEL_OPTIMIZERS_FILE = "model_optimizers.json"


class MissingStepMethodException(Exception):
    pass


class MissingTrainerException(Exception):
    pass


class PyroModule(WithHyperParameters, Callback, nn.Module):
    def __init__(self) -> None:
        super().__init__(hparams_file=MODEL_HPARAMS_FILE)
        self.trainer: Optional["Trainer"] = None
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[Scheduler] = None

    @property
    def device(self):
        return next(self.parameters()).device

    def configure_optimizers(self, loop: "Loop"):
        """
        Define optimizer and optionally scheduler to use during training

        Default values:
        - Optimizer is AdamW with learning rate = trainer.lr
        - Scheduler is None, meaning the learning rate will be constant

        Note:
        Trainer class can be accessed using self.trainer
        If you have multiple optimizers / schedulers, store them in variables.
        >>> You will need to override _fit, and optionally save_optimizers and load_optimizers

        Example:
        ```
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.trainer.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)
        ```
        """
        assert self.trainer is not None, MissingTrainerException(
            "You should assign a trainer to your PyroModule before calling .configure_optimizers"
        )
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.trainer.lr)

    def step(
        self,
        batch: Any,
        stage: Stage,
    ) -> torch.Tensor:
        msg = "A step method must be implemented for your PyroModule model"
        raise MissingStepMethodException(msg)

    def log(self, **data: float | np.ndarray | torch.Tensor):
        assert self.trainer is not None, MissingTrainerException(
            "You should assign a trainer to your PyroModule before calling .log"
        )
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
            self.trainer is not None
            and self.trainer.grad_norm_clip is not None
            and self.trainer.grad_norm_clip > 0.0
        ):
            nn.utils.clip_grad_norm_(self.parameters(), self.trainer.grad_norm_clip)

        # 3. Optimizer step
        if self.optimizer is not None:
            self.optimizer.step()

        # 4. Step the scheduler
        if hasattr(self, "scheduler") and self.scheduler is not None:
            self.scheduler.step()

        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def get_current_lr(self) -> dict[str, float]:
        """
        In the event where you have multiple schedulers, override this method

        Returns:
            dict[str, float]: mapping of learning rate names to their corresponding values
        """
        if hasattr(self, "scheduler") and self.scheduler is not None:
            lr = float(self.scheduler.get_last_lr()[0])
        elif self.trainer is not None:
            lr = self.trainer.lr
        else:
            raise MissingTrainerException(
                "You should assign a scheduler or trainer to your PyroModule before calling .get_current_lr"
            )
        return dict(lr=lr)

    def _is_compiled(self):
        return isinstance(self, torch._dynamo.OptimizedModule)

    def _get_module(self):
        """
        In case the model is compiled, use the _orig_mod attribute instead of the raw model
        """
        return getattr(self, "_orig_mod", self)

    def save_weights(self, folder: Path | str):
        """
        Saves model weights from a specified checkpoint folder.

        Args:
            folder (str): The folder path where the pretrained model is saved.
            file (str): The filename of the file containing the model weights
        """
        folder = Path(folder)
        os.makedirs(folder, exist_ok=True)

        f = folder / MODEL_WEIGHTS_FILE
        log.info(f"Saving model weights to {f}")

        model = self._get_module()
        st.save_model(model=model, filename=f)

    def save_optimizers(self, folder: Path | str):
        if self.optimizer is None:
            warnings.warn(
                "configure_optimizers hasn't been called yet, skipping saving optimizers step"
            )
            return

        folder = Path(folder)
        os.makedirs(folder, exist_ok=True)

        f = folder / MODEL_OPTIMIZERS_FILE
        log.info(f"Saving model optimizers to {f}")

        def save_optim(module: nn.Module):
            return {
                "state_dict": module.state_dict(),
                "class_name": get_classname(module),
            }

        obj = {"optimizer": save_optim(self.optimizer)}
        if hasattr(self, "scheduler"):
            obj["scheduler"] = save_optim(self.scheduler)

        torch.save(obj, f)

    def save(self, folder: Path | str):
        """
        Saves both model's weights and hyperparameters
        """
        self.save_weights(folder=folder)
        self.save_hparams(folder=folder)
        self.save_optimizers(folder=folder)

    def load_weights(
        self,
        folder: Path | str,
        strict: bool = True,
    ):
        """
        Loads model weights from a specified checkpoint folder.

        Args:
            folder (str): The folder path where the pretrained model is saved.
            file (str): The filename of the file containing the model weights
            strict (bool): Whether to strictly enforce that the model weights match the model architecture.
        """
        f = Path(folder) / MODEL_WEIGHTS_FILE
        log.info(f"Loading model weights from {f}")

        model = self._get_module()
        missing, unexpected = st.load_model(model=model, filename=f, strict=strict)
        if not strict:
            warnings.warn(f"Missing layers: {missing}\nUnexpected layers: {unexpected}")

    def load_optimizers(self, folder: Path | str, device: str = "cpu"):
        # TODO: after loading the optimizers, what to do during configure_optimizers?
        # One call => don't care
        # Second call => execute ??
        f = Path(folder) / MODEL_OPTIMIZERS_FILE
        log.info(f"Loading model optimizers from {f}")

        if self.optimizer is not None:
            warnings.warn("Optimizer is already defined, overriding.")

        def load_optim(state: dict[str, dict[str, Any]], key: str, *args):
            s = state[key]
            Class = s["class_name"]
            kwargs = s["state_dict"]
            optim: nn.Module = Class(*args, **kwargs)
            optim.load_state_dict(kwargs)

        state = st.load_file(f, device=device)
        self.optimizer = load_optim(state, "optimizer", self.parameters())
        if hasattr(self, "scheduler"):
            if "scheduler" not in state:
                warnings.warn(
                    "A scheduler is defined in your model, but no scheduler was found when loading the optimizers"
                )
            else:
                self.scheduler = load_optim(state, "scheduler", self.optimizer)

    def load(self, folder: Path | str, strict: bool = True):
        """
        Loads both model's weights and hyperparameters
        """
        self.load_weights(folder=folder, strict=strict)
        self.load_hparams(folder=folder)
        self.load_optimizers(folder=folder)
