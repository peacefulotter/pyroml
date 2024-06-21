import os
import json
import torch
import logging
import safetensors.torch as safetensors

from enum import Enum
from pathlib import Path
from copy import deepcopy
from json import JSONEncoder
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler as Scheduler

from pyroml.config import Config
from pyroml.model import PyroModel
from pyroml.utils import get_date


class EncodeTensor(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super(EncodeTensor, self).default(obj)


log = logging.getLogger(__name__)


class CheckpointFilename(Enum):
    MODEL = "model.safetensors"
    STATE = "training_state.pt"
    PARAM = "hparams.json"


class Checkpoint:
    def __init__(
        self,
        config: Config,
        model: PyroModel,
        optimizer: Optimizer,
        scheduler: Scheduler,
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.date = get_date()
        self.folder: Path | str = None

    def get_cp_folder(self):
        return os.path.join(
            self.config.checkpoint_folder,
            f"{self.date}_{self.config.name}_epoch={self.epoch:03d}_iter={self.iteration:06d}",
        )

    def _get_training_state(self):
        config = deepcopy(self.config)
        # TODO: store metrics value?
        # config.metrics = [type(m).__name__ for m in config.metrics]
        config.device = str(self.device)
        config.optimizer = type(self.optimizer).__name__
        config.scheduler = type(self.scheduler).__name__
        state = {
            "epoch": self.epoch,
            "iteration": self.iteration,
            "config": config.__dict__,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (
                self.scheduler.state_dict() if self.scheduler != None else None
            ),
            "compiled": self._is_model_compiled(),
        }
        return state

    @staticmethod
    def _get_filenames(folder):
        return {fn: os.path.join(folder, fn.value) for fn in CheckpointFilename}

    def _get_folder_and_filenames(self):
        self.folder = self.get_cp_folder()
        os.makedirs(self.folder, exist_ok=True)
        filenames = Checkpoint._get_filenames(self.folder)
        return self.folder, filenames

    def save_model(self):
        folder, filenames = self._get_folder_and_filenames()
        log.info(
            f"Saving model {self.config.name} at epoch {self.epoch}, iter {self.iteration} to {folder}"
        )

        # Saving model weights
        safetensors.save_model(self.model, filenames[CheckpointFilename.MODEL])

        # Saving training state
        tr_state = self._get_training_state()
        torch.save(tr_state, filenames[CheckpointFilename.STATE])

        # Saving model hyperparameters
        # TODO: find a way to store and load hparams
        # with open(filenames[CheckpointFilename.PARAM], "w") as f:
        #    json.dump(self.model.hparams, f, cls=EncodeTensor)

    def _load_state(self, state):
        self.epoch = state["epoch"]
        self.iteration = state["iteration"]
        self.optimizer.load_state_dict(state["optimizer"])
        if self.config.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])

    @staticmethod
    def from_pretrained(
        model: PyroModel = None,
        config: Config = None,
        folder: Path | str = None,
        resume=True,
        strict=True,
        trainer=None,
    ):
        """
        Loads a pretrained model from a specified checkpoint folder.

        Args:
            model (torch.nn.Module): The model to load the pretrained weights into.
            config (Config): The training config.
            folder (str): The folder path where the pretrained model is saved.
            resume (bool, optional): Whether to resume training from the checkpoint. Defaults to True.
            strict (bool, optional): Whether to strictly enforce the shape and type of the loaded weights. Defaults to True.

        Returns:
            Trainer: The trainer object with the pretrained model loaded.
        """
        if trainer is not None:
            model, config, folder = (
                trainer.model,
                trainer.config,
                trainer.folder,
            )
        elif model is None or config is None or folder is None:
            raise ValueError(
                "Either trainer or model, config and folder must be provided"
            )

        log.info(f"Loading checkpoint from {folder}")
        filenames = Checkpoint._get_filenames(folder)

        # Load training state
        state = torch.load(filenames[CheckpointFilename.STATE], map_location="cpu")
        from pyroml.trainer import Trainer

        # Don't use the checkpoint config as it contains erroneous data such as optimizer, scheduler represented as strings
        trainer = trainer or Trainer(model, config)
        if not resume:
            trainer._load_state(state)

        # Load model weights
        # Must be done after creating the trainer if the model has been saved compiled
        # In that case, the model passed as parameter also needs to be compiled before
        missing, unexpected = safetensors.load_model(
            trainer.model,
            filenames[CheckpointFilename.MODEL],
            strict=strict,
        )
        if not strict:
            log.warn(f"Missing layers: {missing}\nUnexpected layers: {unexpected}")

        return trainer
