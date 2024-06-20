import os
import json
import torch
import logging
import safetensors.torch as safetensors

from copy import deepcopy
from json import JSONEncoder


class EncodeTensor(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super(EncodeTensor, self).default(obj)


log = logging.getLogger(__name__)


class Checkpoint:
    MODEL_FILE = "model.safetensors"
    STATE_FILE = "training_state.pt"
    PARAM_FILE = "hparams.json"

    @staticmethod
    def get_checkpoint_path(config, date, epoch, iteration):
        return os.path.join(
            config.checkpoint_folder,
            f"{date}_{config.name}_epoch={epoch:03d}_iter={iteration:06d}",
        )

    def save_model(self):
        config = deepcopy(self.config)
        # TODO: store metrics value?
        config.metrics = [type(m).__name__ for m in config.metrics]
        config.device = str(self.device)
        config.optimizer = type(self.optimizer).__name__
        config.scheduler = type(self.scheduler).__name__
        training_state = {
            "epoch": self.epoch,
            "iter": self.iteration,
            "config": config.__dict__,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (
                self.scheduler.state_dict() if self.scheduler != None else None
            ),
            "compiled": self._is_model_compiled(),
        }

        folder = Checkpoint.get_checkpoint_path(
            self.config, self.date, self.epoch, self.iteration
        )
        os.makedirs(folder, exist_ok=True)

        log.info(
            f"Saving model {self.config.name} at epoch {self.epoch}, iter {self.iteration} to {folder}"
        )
        safetensors.save_model(self.model, os.path.join(folder, Checkpoint.MODEL_FILE))
        torch.save(training_state, os.path.join(folder, Checkpoint.STATE_FILE))
        with open(os.path.join(folder, Checkpoint.PARAM_FILE), "w") as f:
            json.dump(self.model.hparams, f, cls=EncodeTensor)

        return folder

    def _load_state_dict(self, checkpoint, resume):
        if not resume:
            return
        self.epoch = checkpoint["epoch"]
        self.iteration = checkpoint["iter"]
        # self.model.load_state_dict(weights.to(self.device))
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.config.scheduler != None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

    @staticmethod
    def from_pretrained(model, config, folder, resume=True, strict=True):
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
        log.info(f"Loading checkpoint from {folder}")

        # Load checkpoint config
        checkpoint = torch.load(
            os.path.join(folder, Checkpoint.STATE_FILE), map_location="cpu"
        )
        # Don't use the checkpoint config as it contains erroneous data such as optimizer, scheduler represented as strings
        trainer = Trainer(model, config)
        trainer._load_state_dict(checkpoint, resume)

        # Load model weights
        # Required to be done after creating the trainer if the model has been compiled and saved
        # than the model passed as parameter needs to be compiled before as well, and Trainer init makes sure of it
        missing, unexpected = safetensors.load_model(
            trainer.model,
            os.path.join(folder, Checkpoint.MODEL_FILE),
            strict=strict,
        )
        if not strict:
            log.warn(f"Missing layers: {missing}\nUnexpected layers: {unexpected}")

        return trainer
