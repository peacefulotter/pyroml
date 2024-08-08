import os
import re
import torch
import logging
import torch.nn as nn

from pathlib import Path
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler as Scheduler

import pyroml as p

from pyroml.checkpoint import Checkpoint
from pyroml.loop import TrainLoop, TestLoop
from pyroml.log import set_level_all_loggers
from pyroml.hparams import WithHyperParameters
from pyroml.utils import get_classname, get_date

log = logging.getLogger(__name__)


class WrongModuleException(Exception):
    pass


class Trainer(WithHyperParameters):
    # Keep stats of runs from fit (and test?)
    # TODO: log metrics to a txt file too ? do that here or dedicated file logger rather?

    def __init__(
        self,
        loss: nn.Module = nn.MSELoss(),
        lr: float = 1e-4,
        max_epochs: int | None = None,
        max_steps: int | None = 100,
        optimizer: Optimizer = Adam,
        optimizer_params=None,
        scheduler: Scheduler | None = None,
        scheduler_params=None,
        grad_norm_clip: float = 1.0,
        evaluate: bool | str = True,
        evaluate_every: int = 10,
        eval_max_steps: int | None = None,
        device: str | torch.device = "auto",
        dtype: torch.dtype = torch.float32,
        compile: bool = True,
        checkpoint_folder: str | Path = "./checkpoints",
        batch_size: int = 16,
        eval_batch_size: int = None,
        num_workers: int = 4,
        eval_num_workers: int = 0,
        wandb: bool = True,
        wandb_project: str | None = None,
        log_level: int = logging.INFO,
        callbacks: list["p.Callback"] = [],
    ):
        """
        Trainer class specifying all training related parameters and exposing fit / test methods to interact with the model.

        Args:
            loss (torch.nn.Module):
                Loss function.
                Defaults to MSELoss.

            lr (float, optional):
                Learning rate.
                Defaults to 1e-4.

            max_epochs (int, > 0, optional):
                Number of epochs (if max_iterations is not defined).
                Defaults to None.

            max_steps (int, > 0):
                Maximum number of iterations.
                This parameters dominates max_epochs; if number of steps reaches max_steps, training will stop despite max_epochs not being reached.
                Defaults to 100.

            optimizer (torch.optim.Optimizer, optional):
                Optimizer.
                Defaults to Adam.

            optimizer_params (dict, optional):
                Optimizer parameters.
                Defaults to None.

            scheduler (torch.optim.lr_scheduler.LRScheduler, optional):
                Scheduler.
                Defaults to None.

            scheduler_params (dict, optional):
                Scheduler parameters.
                Defaults to None.

            grad_norm_clip (float, optional):
                Gradient norm clipping.
                Defaults to 1.0.

            evaluate (bool or str, optional):
                Whether to periodically evaluate the model on the evaluation dataset, or 'epoch' to evaluate every epoch.
                Defaults to True.

            evaluate_every (int, optional):
                Evaluate every `evaluate_every` iterations / or epoch if evaluate is set to 'epoch'.
                Defaults to 10.

            eval_max_steps (int, optional):
                Maximum number of iterations for the evaluation dataset.
                Defaults to None.

            dtype (torch.dtype, optional):
                Data type to cast model weights to.
                Defaults to torch.float32.

            device (str, optional):
                Device to train on. Defaults to "auto" which will use GPU if available.

            compile (bool, optional):
                Whether to compile the model, this can significantly improve training time but is not supported on all GPUs.
                Defaults to True.

            checkpoint_folder (str, optional):
                Folder to save checkpoints.
                Defaults to "./checkpoints".

            batch_size (int, optional):
                Training batch size. If eval_batch_size is None, evaluation batches will use this value
                Defaults to 16.

            eval_batch_size (int, optional):
                Batch size for the evaluation dataset.
                Defaults to None in which case it will be equal to the training batch size.

            num_workers (int, optional):
                Number of workers for the dataloader.
                Defaults to 4.

            eval_num_workers (int, optional):
                Number of workers for the evaluation dataloader.
                Note that a value > 0 can cause an AssertionError: 'can only test a child process during evaluation'.
                Defaults to 0.

            wandb (bool, optional):
                Whether to use wandb.
                Defaults to True.

            wandb_project (str, optional):
                Wandb project name, if wandb is set to True.
                Defaults to None.

            log_level: (int, optional):
                Logger level. Use logging.X to get the integer corresponding to the level you want
                Defaults to logging.INFO

            callbacks (list[Callback], optional):
                List of callbacks to use.
                Defaults to [].
        Returns:
            Config: Configuration object with the specified hyperparameters.
        """
        super().__init__(hparams_file=Checkpoint.TRAINER_HPARAMS.value)

        scheduler_params = scheduler_params or {}
        optimizer_params = optimizer_params or {}
        eval_batch_size = eval_batch_size or batch_size

        # Training
        self.lr = lr
        self.loss = loss
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.grad_norm_clip = grad_norm_clip

        # Validation
        self.evaluate = evaluate
        self.evaluate_every = evaluate_every
        self.eval_max_steps = eval_max_steps

        # Model
        self.dtype = dtype
        self.device = device
        self.compile = compile
        self.version = 0
        self.checkpoint_folder = Path(checkpoint_folder)
        self._fetch_version()

        # Data
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.eval_num_workers = eval_num_workers

        # Logging
        self.wandb = wandb
        self.wandb_project = wandb_project
        self.log_level = log_level
        set_level_all_loggers(level=self.log_level)

        # Miscs
        self.date = get_date()

        print("TODO: SAVE HPARAMS", dir(self))

        # Callbacks
        self.callbacks = callbacks

        self.model: "p.PyroModel" = None

    def _fetch_version(self):
        for f in os.scandir(self.checkpoint_folder):
            if not f.is_dir() or not re.match(r"v_(\d+)", f.name):
                continue
            version = int(f.name[2:])
            self.version = max(self.version, version)

        self.version += 1
        self.checkpoint_folder = self.checkpoint_folder / f"v_{self.version}"

    def _compile_model(self):
        if self.model._is_compiled():
            log.info("Model is already compiled, skipping compilation")
            return

        log.info(f"Compiling model...")
        self.model = torch.compile(self.model)
        log.info(f"Model compiled!")

    def _setup(self, model: "p.PyroModel"):
        self.model = model

        # Compile model if requested, improves performance
        if self.compile:
            self._compile_model()

        model._setup(self)
        self.model = model

    def _call_loop(
        self, Loop: "p.Loop", model: "p.PyroModel", dataset: Dataset, **kwargs
    ):
        if not isinstance(model, p.PyroModel):
            raise WrongModuleException("Trainer loop model must be a PyroModel")

        self._setup(model)
        loop: "p.Loop" = Loop(model=model, trainer=self, **kwargs)
        loop.run(dataset)
        return loop.tracker.records

    def fit(
        self, model: "p.PyroModel", tr_dataset: Dataset, ev_dataset: Dataset = None
    ):
        return self._call_loop(TrainLoop, model, tr_dataset, ev_dataset=ev_dataset)

    def test(self, model: "p.PyroModel", dataset: Dataset):
        return self._call_loop(TestLoop, model, dataset)

    def save_state(self, folder: Path | str = None):
        # Saving state
        state = {
            "compiled": self.model._is_compiled(),
            "optimizer_name": get_classname(self.optimizer),
            "optimizer": self.optimizer.state_dict(),
        }

        if hasattr(self, "scheduler"):
            state["scheduler_name"] = get_classname(self.scheduler)
            state["scheduler"] = self.scheduler.state_dict()

        folder = Path(folder) or self.checkpoint_folder
        os.makedirs(folder, exist_ok=True)

        f = folder / Checkpoint.TRAINER_STATE.value
        torch.save(obj=state, f=f)

    def load_state(self, folder: Path | str = None):
        folder = Path(folder) or self.checkpoint_folder
        state = torch.load(folder / Checkpoint.TRAINER_STATE.value, map_location="cpu")

        self.optimizer.load_state_dict(state["optimizer"])
        if hasattr(self, "scheduler") and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])

    @staticmethod
    def load(
        self,
        folder: Path | str,
        resume: bool = True,
        strict: bool = True,
    ):
        """
        Loads a pretrained model from a specified checkpoint folder.

        Args:
            folder (str): The folder path where the pretrained model is saved.
            resume (bool, optional): Whether to resume training from the checkpoint. Defaults to True.
            strict (bool, optional): Whether to strictly enforce the shape and type of the loaded weights. Defaults to True.
        """
        folder = Path(folder)
        hparams_file = (
            folder
            if os.path.isfile(folder)
            else folder / Checkpoint.TRAINER_HPARAMS.value
        )
        args = WithHyperParameters._load_hparams(hparams_file)
        print(args)
        trainer = Trainer(**args)
        raise "TODO"
