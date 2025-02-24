import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Type

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pyroml.core.autocast import Autocast
from pyroml.core.hparams import WithHyperParameters
from pyroml.core.model import PyroModule
from pyroml.core.tracker import MetricsTracker
from pyroml.loop import EvalLoop, PredictLoop, TrainLoop
from pyroml.utils import get_classname
from pyroml.utils.log import get_logger, set_level_all_loggers

if TYPE_CHECKING:
    from pyroml.callbacks import Callback
    from pyroml.core.status import Status
    from pyroml.loop.base import Loop

log = get_logger(__name__)

TRAINER_STATE_FILE = "trainer_state.pt"
TRAINER_HPARAMS_FILE = "trainer_hparams.json"


class NotAPyroModuleException(Exception):
    pass


class Trainer(WithHyperParameters):
    def __init__(
        self,
        lr: float = 1e-4,
        max_epochs: Optional[int] = 8,
        max_steps: Optional[int] = None,
        grad_norm_clip: float = 1.0,
        evaluate_on: Literal["step", "epoch", False] = "epoch",
        evaluate_every: int = 1,
        eval_max_steps: Optional[int] = None,
        device: str | torch.device = "auto",
        auto_move: bool = True,
        pin_memory: Optional[bool] = None,
        dtype: torch.dtype = torch.float32,
        compile: bool = False,
        hparams_file: str | Path = TRAINER_HPARAMS_FILE,
        batch_size: int = 16,
        eval_batch_size: Optional[int] = None,
        num_workers: int = 0,
        eval_num_workers: int = 0,
        log_level: int = logging.INFO,
        callbacks: list["Callback"] = [],
    ) -> None:
        """
        Trainer class specifying all training related parameters and exposing fit / test methods to interact with the model.

        Args:
            lr (float, optional):
                Learning rate to use for your optimizer / scheduler
                Defaults to 1e-4.

            max_epochs (int, > 0, optional):
                Number of epochs (if max_iterations is not defined).
                Defaults to 8.

            max_steps (int, > 0):
                Maximum number of iterations.
                *Note:* This parameters dominates max_epochs; if number of steps reaches max_steps, training will stop despite max_epochs not being reached.
                For this reason, we recommend using the loop step related attributes (e.g. loop.total_steps, loop.steps_per_epochs) in your model's
                configure_optimizer method instead of epoch related attributes (e.g. loop.max_epochs)
                Defaults to None.

            grad_norm_clip (float, optional):
                Gradient norm clipping.
                Defaults to 1.0.

            evaluate_on (Literal['step', 'epoch', False], optional):
                Evaluate every `evaluate_every` step or epoch. Setting this to False will disable evaluation.
                Defaults to 'epoch'.

            evaluate_every (int, optional):
                Evaluate model every `evaluate_every` `evaluate_on`. Setting this to a value <= 0 will disable evaluation.
                Defaults to 1.

            eval_max_steps (int, optional):
                Maximum number of iterations for the evaluation dataset. Setting this to a value <= 0 will disable evaluation.
                Defaults to None.

            dtype (torch.dtype, optional):
                Data type to cast model weights to.
                Defaults to torch.float32.

            device (str, optional):
                Device to train on.
                Defaults to "auto" which will use GPU if available.

            auto_move (bool, optional):
                Automatically move batches to device before calling your model's step method. If set to False, your model has to move data to the appropriate device in the step function.
                Defaults to True.

            pin_memory (bool, optional):
                Applies pin_memory to dataloaders. Set this to None to make pin_memory true only if training on gpu.
                Default to None.

            compile (bool, optional):
                Whether to compile the model, this can significantly improve inference time but is not supported on all GPUs.
                Defaults to False.

            hparams_file (str, optional):
                File to save hyperparameters.
                Defaults to Checkpoint.TRAINER_HPARAMS.value.

            batch_size (int, optional):
                Training batch size. If eval_batch_size is None, evaluation batches will use this value.
                Defaults to 16.

            eval_batch_size (int, optional):
                Batch size for the evaluation dataset. Setting this to a value <= 0 will disable evaluation.
                Defaults to None in which case it will be equal to the training batch size.

            num_workers (int, optional):
                Number of workers for the dataloader.
                Defaults to 0.

            eval_num_workers (int, optional):
                Number of workers for the evaluation dataloader.
                Note that a value > 0 can cause an AssertionError: 'can only test a child process during evaluation'.
                Defaults to 0.

            log_level (int, optional):
                Logger level. Use logging.X to get the integer corresponding to the level you want
                Defaults to logging.INFO

            callbacks (list[Callback], optional):
                List of callbacks to use.
                Defaults to [].
        Returns:
            Config: Configuration object with the specified hyperparameters.
        """
        super().__init__(hparams_file=hparams_file)

        # Training
        self.lr = lr
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.grad_norm_clip = grad_norm_clip
        self.callbacks = callbacks

        # Validation
        assert evaluate_on == "step" or evaluate_on == "epoch" or evaluate_on is False
        self.evaluate_on = evaluate_on
        self.evaluate_every = evaluate_every
        self.eval_max_steps = eval_max_steps

        # Model
        self.dtype = dtype
        self.device = device
        self.compile = compile
        self.version = -1
        self.model: Optional["PyroModule" | torch._dynamo.OptimizedModule] = None

        # Data
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.num_workers = num_workers
        self.eval_num_workers = eval_num_workers

        # Logging
        self.log_level = log_level

        # Device and dtypes
        self.autocast: Autocast = Autocast(self)
        self.auto_move: bool = auto_move
        self.pin_memory: Optional[bool] = pin_memory

        self._status_stack: list["Status"] = []

        self.tracker: MetricsTracker = MetricsTracker()

    @property
    def current_status(self) -> Optional["Status"]:
        return self._status_stack[-1] if len(self._status_stack) > 0 else None

    @property
    def eval_enabled(self):
        return (
            self.evaluate_on is not False
            and self.evaluate_every > 0
            and (self.eval_max_steps is None or self.eval_max_steps > 0)
            and (self.eval_batch_size is None or self.eval_batch_size > 0)
        )

    def log(self, **data: float | np.ndarray | torch.Tensor):
        return self.tracker.log(**data)

    @property
    def log_level(self):
        return self._log_level

    @log_level.setter
    def log_level(self, level):
        set_level_all_loggers(level=level)
        self._log_level = level

    def _setup_model(self, model: "PyroModule"):
        self.model = model

        # Compile model if requested, should improve performance for inference
        if self.compile:
            self.model.compile()

        self.model._setup(self)

    def _call_loop(
        self, Loop: Type["Loop"], model: "PyroModule", dataset: Dataset, **kwargs
    ) -> pd.DataFrame:
        """ "
        Setups the trainer and model and
        Calls a loop with the specified model and dataset.
        """
        if not isinstance(model, PyroModule) or (
            isinstance(model, torch._dynamo.OptimizedModule)
            and isinstance(model._orig_mod, PyroModule)
        ):
            raise NotAPyroModuleException(
                "Trainer runs loops with a model that must be a PyroModule"
            )

        loop = Loop(model=model, trainer=self, dataset=dataset, **kwargs)
        self._setup_model(model)
        self._status_stack.append(loop)
        res = loop.run()
        self._status_stack.pop()
        return res

    def fit(
        self,
        model: "PyroModule",
        tr_dataset: Dataset,
        ev_dataset: Optional[Dataset] = None,
    ) -> "MetricsTracker":
        return self._call_loop(
            Loop=TrainLoop, model=model, dataset=tr_dataset, ev_dataset=ev_dataset
        )

    def evaluate(self, model: "PyroModule", dataset: Dataset) -> "MetricsTracker":
        return self._call_loop(Loop=EvalLoop, model=model, dataset=dataset)

    def predict(
        self, model: "PyroModule", dataset: Dataset
    ) -> tuple["MetricsTracker", Any]:
        return self._call_loop(Loop=PredictLoop, model=model, dataset=dataset)

    def save(
        self,
        folder: Path | str,
        file: Path | str = TRAINER_STATE_FILE,
    ) -> None:
        # Saving state
        state = {}

        if self.model is not None:
            state["compiled"] = self.model._is_compiled()

            if self.model.optimizer is not None:
                state.update(
                    {
                        "optimizer_name": get_classname(self.model.optimizer),
                        "optimizer": self.model.optimizer.state_dict(),
                    }
                )

            if hasattr(self.model, "scheduler") and self.model.scheduler is not None:
                state["scheduler_name"] = get_classname(self.model.scheduler)
                state["scheduler"] = self.model.scheduler.state_dict()

        folder = Path(folder)
        os.makedirs(folder, exist_ok=True)
        torch.save(obj=state, f=folder / file)

    @staticmethod
    def load(
        folder: Path | str,
        file: Path | str = TRAINER_HPARAMS_FILE,
        model: Optional["PyroModule"] = None,
    ):
        # TODO: two methods, one for loading the trainer
        # Another one for continuint training ?
        """
        Loads a pretrained model from a specified checkpoint folder.

        Args:
            folder (str): The folder path where the pretrained model is saved.
            resume (bool, optional): Whether to resume training from the checkpoint. Defaults to True.
            strict (bool, optional): Whether to strictly enforce the shape and type of the loaded weights. Defaults to True.
        """
        """
        Loads a trainer state from a specified checkpoint folder.

        Args:
            folder (str): The folder path where the pretrained model is saved.
        """
        folder = Path(folder)
        args = WithHyperParameters._load_hparams(folder / file)
        trainer = Trainer(**args)

        # TODO: find a way to reload the optimizer and scheduler to the model
        if model is not None:
            model.load

        # optim = getattr(torch.optim, args.optimizer)
        # sched = getattr(torch.optim.lr_scheduler, args.scheduler)
        # Should be loaded in the model
        print(args)

        # TODO: callbacks state saved with WithHyperparameters
        warnings.warn(
            "Loading callbacks is not supported, if you have custom callbacks please add them to the trainer again"
        )

        return trainer
