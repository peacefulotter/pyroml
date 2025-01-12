import os
import re
import torch
import logging
import warnings
import pandas as pd
import torch.nn as nn

from pathlib import Path
from torch.utils.data import Dataset


import pyroml as p

from pyroml.loop.autocast import Autocast
from pyroml.checkpoint import Checkpoint
from pyroml.log import initialize_logging
from pyroml.loop import TrainLoop, TestLoop
from pyroml.log import set_level_all_loggers
from pyroml.hparams import WithHyperParameters
from pyroml.utils import get_classname, get_date

log = p.get_logger(__name__)


class WrongModuleException(Exception):
    pass


class Trainer(WithHyperParameters):
    def __init__(
        self,
        loss: nn.Module = nn.MSELoss(),
        lr: float = 1e-4,
        max_epochs: int | None = None,
        max_steps: int | None = 100,
        grad_norm_clip: float = 1.0,
        evaluate: bool | str = True,
        evaluate_every: int = 10,
        eval_max_steps: int | None = None,
        device: str | torch.device = "auto",
        dtype: torch.dtype = torch.float32,
        compile: bool = True,
        checkpoint_folder: str | Path = "./checkpoints",
        hparams_file: str | Path = Checkpoint.TRAINER_HPARAMS.value,
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
                Learning rate to use for your optimizer / scheduler
                Defaults to 1e-4.

            max_epochs (int, > 0, optional):
                Number of epochs (if max_iterations is not defined).
                Defaults to None.

            max_steps (int, > 0):
                Maximum number of iterations.
                This parameters dominates max_epochs; if number of steps reaches max_steps, training will stop despite max_epochs not being reached.
                Defaults to 100.

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

            hparams_file (str, optional):
                File to save hyperparameters.
                Defaults to Checkpoint.TRAINER_HPARAMS.value.

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

        eval_batch_size = eval_batch_size or batch_size

        # Training
        self.lr = lr
        self.loss = loss
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.grad_norm_clip = grad_norm_clip
        self.callbacks = callbacks

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
        self.model: "p.PyroModel" | torch._dynamo.OptimizedModule = None

        # Data
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.eval_num_workers = eval_num_workers

        # Logging
        self.wandb = wandb
        self.wandb_project = self._get_wandb_project(wandb, wandb_project)
        self.log_level = log_level

        self._setup()

    def _get_wandb_project(self, wandb: bool, wandb_project: str | None):
        project = wandb_project or os.environ.get("WANDB_PROJECT")
        if wandb and (project == "" or project is None):
            raise ValueError(
                "Wandb project name is required, please set WANDB_PROJECT in your environment variables or pass wandb_project in the Trainer constructor"
            )
        return project

    def _fetch_version(self):
        os.makedirs(self.checkpoint_folder, exist_ok=True)
        for f in os.scandir(self.checkpoint_folder):
            if not f.is_dir() or not re.match(r"v_(\d+)", f.name):
                continue
            version = int(f.name[2:])
            self.version = max(self.version, version)

        self.version += 1
        self.checkpoint_folder = self.checkpoint_folder / f"v_{self.version}"

    def _get_total_nb_steps(self, dataset: Dataset) -> int:
        if self.max_steps:
            return self.max_steps
        if self.max_epochs:
            return self.max_epochs * len(dataset) // self.batch_size
        raise ValueError("Trainer max_steps or max_epochs must be defined")

    def _setup(self):
        self.autocast = Autocast(self)
        self.date = get_date()
        initialize_logging()
        set_level_all_loggers(level=self.log_level)
        self._fetch_version()

    def _setup_model(self, model: "p.PyroModel", dataset: Dataset):
        self.model = model

        # Compile model if requested, improves performance
        if self.compile:
            self.model.compile()

        self.model._setup(self)

    def _call_loop(
        self, Loop: "p.Loop", model: "p.PyroModel", dataset: Dataset, **kwargs
    ) -> pd.DataFrame:
        """ "
        Setups the trainer and model and
        Calls a loop with the specified model and dataset.
        """
        if not isinstance(model, p.PyroModel) and (
            not isinstance(model, torch._dynamo.OptimizedModule)
            or not isinstance(model._orig_mod, p.PyroModel)
        ):
            raise WrongModuleException("Trainer loop model must be a PyroModel")

        self._setup()
        self._setup_model(model, dataset)
        loop: "p.Loop" = Loop(model=model, trainer=self, **kwargs)
        loop.run(dataset)
        return loop.tracker

    def fit(
        self, model: "p.PyroModel", tr_dataset: Dataset, ev_dataset: Dataset = None
    ) -> "p.MetricTracker":
        return self._call_loop(TrainLoop, model, tr_dataset, ev_dataset=ev_dataset)

    def test(self, model: "p.PyroModel", dataset: Dataset) -> "p.MetricTracker":
        return self._call_loop(TestLoop, model, dataset)

    def save_state(
        self,
        folder: Path | str = None,
        file: Path | str = Checkpoint.TRAINER_STATE.value,
    ) -> None:
        # Saving state
        state = {
            "compiled": self.model._is_compiled(),
            "optimizer_name": get_classname(self.model.optimizer),
            "optimizer": self.model.optimizer.state_dict(),
        }

        if hasattr(self.model, "scheduler"):
            state["scheduler_name"] = get_classname(self.model.scheduler)
            state["scheduler"] = self.model.scheduler.state_dict()

        folder = Path(folder) or self.checkpoint_folder
        os.makedirs(folder, exist_ok=True)
        torch.save(obj=state, f=folder / file)

    def load_state(
        self,
        folder: Path | str = None,
        file: Path | str = Checkpoint.TRAINER_STATE.value,
    ) -> None:
        folder = Path(folder) or self.checkpoint_folder
        state = torch.load(folder / file, map_location="cpu")

        self.model.optimizer.load_state_dict(state["optimizer"])
        if hasattr(self, "scheduler") and "scheduler" in state:
            self.model.scheduler.load_state_dict(state["scheduler"])

    @staticmethod
    def load(
        folder: Path | str,
        file: Path | str = Checkpoint.TRAINER_HPARAMS.value,
    ):
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
        hparams_file = folder if os.path.isfile(folder) else folder / file
        args = WithHyperParameters._load_hparams(hparams_file)

        # Resolve loss, optimizer and scheduler instances from name
        args.loss = getattr(nn, args.loss)()

        # TODO: find a way to reload the optimizer and scheduler to the model
        # optim = getattr(torch.optim, args.optimizer)
        # sched = getattr(torch.optim.lr_scheduler, args.scheduler)
        print(args)

        warnings.warn(
            "Loading callbacks is not supported, if you have custom callbacks please add them to the trainer again"
        )

        trainer = Trainer(**args)
        return trainer
