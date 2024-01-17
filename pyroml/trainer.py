import os
import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader, RandomSampler
import safetensors.torch as safetensors

from .wandb import Wandb
from .stats import Statistics
from .utils import to_device, get_date, unwrap_model, Callbacks
from .logger import Logger


class Trainer(Callbacks):
    MODEL_FILE = "model.safetensors"
    STATE_FILE = "state.pt"

    def __init__(self, model, config):
        self.config = config
        self.logger = Logger("Trainer", config)
        self.date = get_date()

        self.device = torch.device("cpu")
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.log(f"Set on device {self.device}", force=True)

        self.model = model.to(device=self.device)
        self.compile_model()

        self.epoch = 0
        self.iteration = 0
        self.optimizer = self.config.optimizer(
            self.model.parameters(), lr=self.config.lr, **self.config.optimizer_params
        )
        self.criterion = self.config.criterion()
        self.scheduler = None
        if self.config.scheduler:
            self.scheduler = self.config.scheduler(
                self.optimizer, **self.config.scheduler_params
            )

        if config.wandb:
            self.wandb = Wandb(config)

        Callbacks.__init__(self)

    def compile_model(self):
        if not self.config.compile:
            return

        if isinstance(self.model, torch._dynamo.OptimizedModule):
            self.logger.log(
                "Skipping compilation because given model is already compiled"
            )
            return

        self.logger.log(f"Compiling model...", force=True)
        self.model = torch.compile(self.model)
        self.logger.log(f"Model compiled!", force=True)

    @staticmethod
    def get_checkpoint_path(config, date, epoch, iteration):
        return os.path.join(
            config.checkpoint_folder,
            f"{date}_{config.name}_epoch={epoch:03d}_iter={iteration:06d}",
        )

    def save_model(self):
        config = deepcopy(self.config)
        config.metrics = [type(m).__name__ for m in config.metrics]
        config.device = str(self.device)
        config.optimizer = type(self.optimizer).__name__
        config.criterion = type(self.criterion).__name__
        config.scheduler = type(self.scheduler).__name__
        state = {
            "epoch": self.epoch,
            "iter": self.iteration,
            "config": config.__dict__,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
            if self.scheduler != None
            else None,
        }

        folder = Trainer.get_checkpoint_path(
            self.config, self.date, self.epoch, self.iteration
        )
        if not os.path.exists(folder):
            os.makedirs(folder)

        self.logger.log(
            f"Saving model {self.config.name} at epoch {self.epoch}, iter {self.iteration} to {folder}",
            force=True,
        )
        safetensors.save_model(self.model, os.path.join(folder, Trainer.MODEL_FILE))
        torch.save(state, os.path.join(folder, Trainer.STATE_FILE))
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
        Loads a pretrained model from a specified folder.

        Args:
            model (torch.nn.Module): The model to load the pretrained weights into.
            config (Config): The training config.
            folder (str): The folder path where the pretrained model is saved.
            resume (bool, optional): Whether to resume training from the checkpoint. Defaults to True.
            strict (bool, optional): Whether to strictly enforce the shape and type of the loaded weights. Defaults to True.

        Returns:
            Trainer: The trainer object with the pretrained model loaded.
        """
        logger = Logger("Trainer", config)
        logger.log("Loading checkpoint from", folder, force=True)

        # Load checkpoint config
        checkpoint = torch.load(
            os.path.join(folder, Trainer.STATE_FILE), map_location="cpu"
        )
        # Don't use the checkpoint config as it contains erroneous data such as optimizer, scheduler represented as strings
        # Moreover, this allows to change the config between runs, even tho this is already possible from one training to another
        trainer = Trainer(model, config)
        trainer._load_state_dict(checkpoint, resume)

        # Load model weights
        # Required to be done after creating the trainer if the model has been compiled and saved
        # than the model passed as parameter needs to be compiled before as well, and Trainer init makes sure of it
        missing, unexpected = safetensors.load_model(
            trainer.model,
            os.path.join(folder, Trainer.MODEL_FILE),
            strict=strict,
        )
        if not strict:
            logger.log(
                "Loading weights: missing",
                missing,
                ", unexpected",
                unexpected,
                force=True,
            )

        return trainer

    def on_batch_end(self, statistics, output, target, loss):
        self.trigger_callbacks("on_batch_end")

        if self.config.stats_every == None or (
            self.iteration != 0 and self.iteration % self.config.stats_every != 0
        ):
            return

        stats = statistics.register(output, target, loss, self.epoch, self.iteration)

        if self.config.wandb:
            self.wandb.log(stats)

        self.trigger_callbacks("on_stats", **stats)

    def fit(self, train_dataset, eval_dataset=None):
        self.model.train()
        self.date = get_date()

        statistics = Statistics(
            self.model,
            self.criterion,
            self.scheduler,
            self.config,
            self.device,
            eval_dataset,
        )

        train_loader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset, replacement=True),
            shuffle=False,
            pin_memory=self.device != "cpu",
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )
        data_iter = iter(train_loader)

        if self.config.wandb:
            self.logger.log("Initializing wandb")
            self.wandb.init(self.model, self.optimizer, self.criterion, self.scheduler)

        self.logger.log("Starting training")
        while self.iteration < self.config.max_iterations and (
            self.config.epochs == None or self.epoch < self.config.epochs
        ):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                self.epoch += 1
                self.trigger_callbacks("on_epoch_end")
                batch = next(data_iter)

            self.trigger_callbacks("on_batch_start")
            data, target = batch
            data, target = to_device(data, self.device), to_device(target, self.device)

            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            if self.config.grad_norm_clip != None and self.config.grad_norm_clip != 0.0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_norm_clip
                )
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.on_batch_end(statistics, output, target, loss)

            self.iteration += 1

        cp_path = self.save_model()
        return statistics, cp_path
