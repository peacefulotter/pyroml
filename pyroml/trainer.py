import os
import torch
import logging
import torch.nn as nn

from torch.utils.data import DataLoader, RandomSampler

from pyroml.wandb import Wandb
from pyroml.config import Config
from pyroml.progress import Progress
from pyroml.checkpoint import Checkpoint
from pyroml.utils import to_device, get_date, Callbacks, Stage

log = logging.getLogger(__name__)


class Trainer(Checkpoint, Callbacks):
    def __init__(self, model: nn.Module, config: Config):
        self.model = model
        self.config = config

        if self.config.debug:
            log.setLevel(logging.DEBUG)

        self.date = get_date()

        # Device selection, auto will default to cuda if available
        self.device = torch.device("cpu")
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device {self.device}")

        # Compile model if requested, improves performance
        if self.config.compile:
            self.compile_model()

        # Training setup
        self.epoch = 0
        self.iteration = 0
        self.optimizer = self.config.optimizer(
            self.model.parameters(), lr=self.config.lr, **self.config.optimizer_params
        )
        self.scheduler = None
        if self.config.scheduler:
            self.scheduler = self.config.scheduler(
                self.optimizer, **self.config.scheduler_params
            )

        if config.wandb:
            self.wandb = Wandb(config)

        self.train_loader: DataLoader = None
        self.progress: Progress = Progress()

        Callbacks.__init__(self)

    def _is_model_compiled(self):
        return isinstance(self.model, torch._dynamo.OptimizedModule)

    def compile_model(self):
        if self._is_model_compiled():
            log.info("Model is already compiled, skipping compilation")
            return

        log.info(f"Compiling model...")
        self.model = torch.compile(self.model)
        log.info(f"Model compiled!")

    # TODO: logs and statistics
    # TODO: progress bar should reflects logged metrics
    """def _log(self, ...):
        
        if self.config.wandb:
            self.wandb.log(stats)"""

    def _get_batch(self, data_iter):
        try:
            batch = next(data_iter)
        except StopIteration:
            # TODO: new progress bar on new epoch
            self.trigger_callbacks("on_train_epoch_end")
            data_iter = iter(self.train_loader)
            self.epoch += 1
            self.progress.update(self.training_task, description=f"Epoch {self.epoch}")

            batch = next(data_iter)

        return to_device(batch, self.device)

    def _extract_loss(self, output):
        if isinstance(output, dict):
            return self._extract_loss(output["loss"])
        elif isinstance(output, tuple) or isinstance(output, list):
            return self._extract_loss(output[0])
        elif isinstance(output, torch.Tensor):
            return output
        else:
            raise ValueError(
                "The return type of the model.step function should be a torch.Tensor, dict[torch.Tensor], or tuple[torch.Tensor]"
                + "\nIn the case of returning a list or tuple, make sure the loss is a torch.Tensor located at the first position"
            )

    def _fit_model(self, loss):
        """
        - Backpropagate loss
        - Clip the gradients if necessary
        - Step the optimizer
        - Step the scheduler if available
        - Zero the gradients
        """
        loss.backward()
        if self.config.grad_norm_clip != None and self.config.grad_norm_clip != 0.0:
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_norm_clip
            )
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _training_loop(self):
        train_loader_iter = iter(self.train_loader)

        while self.iteration < self.config.max_iterations and (
            self.config.epochs == None or self.epoch < self.config.epochs
        ):
            batch = self._get_batch(train_loader_iter)

            self.trigger_callbacks("on_train_iter_start")

            output = self.model.step(batch, Stage.TRAIN)
            loss = self._extract_loss(output)
            self._fit_model(loss)

            self.trigger_callbacks("on_train_iter_end")

            self.progress.update(trainning_task_id, advance=1)
            self.iteration += 1

    def fit(self, train_dataset, eval_dataset=None):
        self.date = get_date()

        self.model.train()
        self.model.to(device=self.device)

        self.train_loader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset, replacement=True),
            shuffle=False,
            pin_memory=self.device != "cpu",
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

        if self.config.wandb:
            self.wandb.init(self.model, self.optimizer, self.scheduler)

        with self._get_progress_bar(Stage.TRAIN) as progress:
            self.progress = progress
            self._training_loop()

        self.cp_path = self.save_model()

        self.model.eval()
        self.model.cpu()
