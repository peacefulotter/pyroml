import os
import torch
import logging
import torch.nn as nn

from torch.utils.data import DataLoader, RandomSampler

from pyroml.wandb import Wandb
from pyroml.config import Config
from pyroml.tracker import Tracker
from pyroml.model import PyroModel
from pyroml.progress import Progress
from pyroml.checkpoint import Checkpoint
from pyroml.utils import to_device, get_date, Callback, CallbackHandler, Stage

log = logging.getLogger(__name__)


class Trainer(Checkpoint, CallbackHandler):
    def __init__(self, model: PyroModel, config: Config):
        self.model = model
        self.config = config

        if self.config.debug:
            log.setLevel(logging.DEBUG)

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

        self.tracker = Tracker()

        self.tr_loader: DataLoader = None
        self.ev_loader: DataLoader = None
        self.progress: Progress = None

        CallbackHandler.__init__(self)
        Checkpoint.__init__(
            self, self.config, self.model, self.optimizer, self.scheduler
        )

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

    def _get_batch(self, data_loader, data_iter):
        try:
            batch = next(data_iter)
        except StopIteration:
            self.trigger_callbacks(Callback.ON_TRAIN_EPOCH_END)
            data_iter = iter(data_loader)
            batch = next(data_iter)
            self.epoch += 1

        import time

        time.sleep(0.3)

        batch = to_device(batch, self.device)
        return data_iter, batch

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

    def _start_epoch(self, stage, loader):
        self.progress.set_stage(stage, loader)
        cb = (
            Callback.ON_TRAIN_EPOCH_START
            if stage == Stage.TRAIN
            else Callback.ON_VAL_EPOCH_START
        )
        self.trigger_callbacks(cb)

    @torch.no_grad()
    def _validation_loop(self):
        self.model.eval()

        self._start_epoch(Stage.VAL, self.ev_loader)
        ev_loader_iter = iter(self.ev_loader)

        iterations = 0
        while iterations < len(self.ev_loader) and (
            self.config.eval_max_iterations is None
            or iterations < self.config.eval_max_iterations
        ):
            _, batch = self._get_batch(self.ev_loader, ev_loader_iter)

            self.trigger_callbacks(Callback.ON_VAL_ITER_START)

            output = self.model.step(batch, Stage.VAL)

            self.trigger_callbacks(Callback.ON_VAL_ITER_END)

            last_metrics = self.tracker.update(Stage.VAL, self.epoch, output)
            self.progress.advance(metrics=last_metrics)
            iterations += 1

        self.trigger_callbacks(Callback.ON_VAL_EPOCH_END)

        self.progress.set_stage(Stage.TRAIN)
        self.model.train()

    def _training_loop(self):
        self.progress.new_epoch(self.epoch)
        self._start_epoch(Stage.TRAIN, self.tr_loader)
        tr_loader_iter = iter(self.tr_loader)

        while self.iteration < self.config.max_iterations:
            if (
                self.config.evaluate
                and self.iteration % self.config.evaluate_every == 0
            ):
                self._validation_loop()

            # Retrieve a new batch and memorize the epoch
            old_epoch = self.epoch
            tr_loader_iter, batch = self._get_batch(self.tr_loader, tr_loader_iter)

            # In case the epoch has changed, either move on to the new epoch or exit if max_epochs is reached
            if old_epoch != self.epoch:
                if (
                    self.config.max_epochs is None
                    or self.epoch < self.config.max_epochs
                ):
                    self.progress.new_epoch(self.epoch)
                    tr_loader_iter = self._start_epoch(Stage.TRAIN, self.tr_loader)
                else:
                    break

            self.trigger_callbacks(Callback.ON_TRAIN_ITER_START)

            output = self.model.step(batch, Stage.TRAIN)

            loss = self._extract_loss(output)
            self._fit_model(loss)

            self.trigger_callbacks(Callback.ON_TRAIN_ITER_END)

            last_metrics = self.tracker.update(Stage.TRAIN, self.epoch, output)
            self.progress.advance(metrics=last_metrics)
            self.iteration += 1

    def fit(self, tr_dataset, ev_dataset=None):
        if self.config.evaluate and ev_dataset is None:
            log.warn(
                "You have chosen to evaluate the model, but no evaluation dataset is passed. Ignoring evaluation."
            )

        self.date = get_date()

        self.model.train()
        self.model.to(device=self.device)

        self.tr_loader = DataLoader(
            tr_dataset,
            sampler=RandomSampler(tr_dataset, replacement=True),
            shuffle=False,
            pin_memory=str(self.device) != "cpu",
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

        self.ev_loader = (
            DataLoader(
                ev_dataset,
                shuffle=False,
                pin_memory=self.device != "cpu",
                batch_size=self.config.eval_batch_size,
                num_workers=self.config.eval_num_workers,
            )
            if ev_dataset is not None
            else None
        )

        if self.config.wandb:
            self.wandb.init(self.model, self.optimizer, self.scheduler)

        self.progress = Progress()
        with self.progress.bar:
            self._training_loop()

        self.save_model()
        self.model.eval()
        self.model.cpu()

    def test(self, dataset):
        self.model.eval()
        self.model.to(device=self.device)

        progress = Progress()
        tracker = Tracker()

        te_loader = DataLoader(
            dataset,
            shuffle=False,
            pin_memory=self.device != "cpu",
            batch_size=self.config.eval_batch_size,
            num_workers=self.config.eval_num_workers,
        )

        progress.set_stage(Stage.TEST, te_loader)
        te_loader_iter = iter(te_loader)

        self.trigger_callbacks(Callback.ON_TEST_EPOCH_START)

        with progress.bar:
            while True:
                old_epoch = self.epoch
                te_loader_iter, batch = self._get_batch(te_loader, te_loader_iter)

                if old_epoch != self.epoch:
                    break

                self.trigger_callbacks(Callback.ON_TEST_ITER_START)

                output = self.model.step(batch, Stage.TEST)

                self.trigger_callbacks(Callback.ON_TEST_ITER_END)

                last_metrics = tracker.update(Stage.TEST, 0, output)
                progress.advance(metrics=last_metrics)

        self.trigger_callbacks(Callback.ON_TEST_EPOCH_END)

        self.model.cpu()

        return tracker
