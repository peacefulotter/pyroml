import torch
import logging
import torch.nn as nn
import safetensors.torch as st

from enum import Enum
from pathlib import Path
from typing import TypeAlias
from torchmetrics import Metric
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler

from pyroml.config import Config
from pyroml.checkpoint import Checkpoint
from pyroml.utils import Stage, __classname

log = logging.getLogger(__name__)


class Step(Enum):
    PRED = "pred"
    METRIC = "metric"
    TARGET = "target"


StepData: TypeAlias = torch.Tensor | dict[torch.Tensor] | tuple[torch.Tensor]
StepOutput: TypeAlias = dict[Step, torch.Tensor]


class MissingStepMethodException(Exception):
    pass


class PyroModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config: Config
        self.optimizer: Optimizer
        self.scheduler: Scheduler | None

    def step(
        self,
        batch: StepData,
        stage: Stage,
    ) -> StepOutput:
        msg = "a step method must be implemented for your PyroModel model"
        raise MissingStepMethodException(msg)

    def configure_metrics(
        self,
    ) -> dict[Metric] | None:
        pass

    def configure_optimizers(self, config: Config):
        self.config = config

        self.optimizer: Optimizer = config.optimizer(
            self.parameters(), lr=config.lr, **config.optimizer_params
        )

        self.scheduler: Scheduler | None = None
        if config.scheduler is not None:
            self.scheduler = config.scheduler(self.optimizer, **config.scheduler_params)

    def _fit(self, output: StepOutput):
        """
        Perform a training step on the model using the output of the step method.
        Override this method if you wish to customize the training loop.

        1. Computes loss based on config.loss
        2. Computes gradient using loss.backward()
        3. Clip the gradients if necessary
        4. Step the optimizer, effectively backpropagating the loss
        5. Step the scheduler if available
        6. Zero the gradients
        """
        # Compute the loss
        pred = output[Step.PRED]
        target = output[Step.TARGET]
        loss: torch.Tensor = self.config.loss(pred, target)

        # Compute gradients
        loss.backward()

        # Clip the gradients
        if self.config.grad_norm_clip != None and self.config.grad_norm_clip != 0.0:
            nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_norm_clip)

        # Backpropagate the loss
        self.optimizer.step()

        # Step the scheduler
        if self.scheduler:
            self.scheduler.step()

        # Zero the gradients
        self.optimizer.zero_grad(set_to_none=True)

        output["loss"] = loss.item()

    def get_current_lr(self):
        if self.scheduler == None:
            return self.config.lr
        return float(self.scheduler.get_last_lr()[0])

    # TODO !!!: even possible to return true in nn.Module?
    # Make sure the trainer is using the compiled version if requested
    def _is_compiled(self):
        return isinstance(self, torch._dynamo.OptimizedModule)

    def save(self, folder: Path):
        log.info(f"Saving model to {folder}")

        # Saving model weights
        st.save_model(self.model, folder / Checkpoint.MODEL_WEIGHTS)

        # Saving state
        # TODO: move this to trainer state file
        state = self._get_model_state()
        torch.save(state, folder / Checkpoint.MODEL_STATE)

        # Saving model hyperparameters
        # TODO: find a way to store and load hparams
        # with open(folder / Checkpoint.MODEL_HPARAMS, "w") as f:
        #    json.dump(self.model.hparams, f, cls=EncodeTensor)

    def _load_state(self, state):
        self.epoch = state["epoch"]
        self.iteration = state["iteration"]
        self.optimizer.load_state_dict(state["optimizer"])
        if hasattr(self, "scheduler") and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])

    def _get_model_state(self):
        state = {
            "compiled": self._is_compiled(),
            "optimizer_name": __classname(self.optimizer),
            "optimizer": self.optimizer.state_dict(),
        }
        if hasattr(self, "scheduler"):
            state["scheduler_name"] = __classname(self.scheduler)
            state["scheduler"] = self.scheduler.state_dict()

        return state

    def from_pretrained(self, folder: Path = None, strict: bool = True):
        """
        Loads a pretrained model from a specified checkpoint folder.

        Args:
            folder (str): The folder path where the pretrained model is saved.
            strict (bool): Whether to strictly enforce that the model weights match the model architecture.
        """
        log.info(f"Loading checkpoint from {folder}")

        # Load training state
        state = torch.load(folder / Checkpoint.MODEL_STATE, map_location="cpu")
        self._load_state(state)

        # Load model weights
        # Must be done after creating the trainer if the model has been saved compiled
        # In that case, the model passed as parameter also needs to be compiled before
        missing, unexpected = st.load_model(
            model=self, filename=folder / Checkpoint.MODEL_WEIGHTS, strict=strict
        )
        if not strict:
            log.warn(f"Missing layers: {missing}\nUnexpected layers: {unexpected}")
