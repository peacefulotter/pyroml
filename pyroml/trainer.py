import os
import torch
import logging
import torch.nn as nn

from typing import Callable
from torchmetrics import Metric
from torch.utils.data import DataLoader, RandomSampler

from pyroml.wandb import Wandb
from pyroml.config import Config
from pyroml.tracker import MetricsTracker
from pyroml.progress import Progress
from pyroml.checkpoint import Checkpoint
from pyroml.model import PyroModel, StepOutput, Step
from pyroml.utils import to_device, get_date, Callback, CallbackHandler, Stage

log = logging.getLogger(__name__)


class Trainer(Checkpoint, CallbackHandler):
    def __init__(self, model: PyroModel, config: Config):
        self.model = model
        self.config = config

        if self.config.verbose:
            log.setLevel(logging.INFO)
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

        self.tracker = MetricsTracker()
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

    def _get_batch(
        self,
        stage: Stage,
        data_loader: DataLoader,
        data_iter,
        progress: Progress,
        epoch: int,
        max_epochs: int,
    ):
        try:
            batch = next(data_iter)
        except StopIteration:
            self.trigger_callback(Callback.ON_EPOCH_END(stage))

            if max_epochs > 0 and epoch + 1 >= max_epochs:
                return None, None, None

            progress.add_stage(stage, data_loader)

            data_iter = iter(data_loader)
            batch = next(data_iter)
            epoch += 1

        batch = to_device(batch, self.device)

        return data_iter, batch, epoch

    def _extract_loss(self, output):
        if isinstance(output, dict):
            return self._extract_loss(output["loss"])
        elif isinstance(output, torch.Tensor):
            return output
        else:
            raise ValueError(
                "The return type of the model.step function should be a torch.Tensor, or dict[torch.Tensor]"
            )

    def _fit_model(self, output: StepOutput):
        """
        - Backpropagate loss
        - Clip the gradients if necessary
        - Step the optimizer
        - Step the scheduler if available
        - Zero the gradients
        """
        pred = output[Step.PRED]
        target = output[Step.TARGET]
        loss = self.config.loss(pred, target)

        loss.backward()
        if self.config.grad_norm_clip != None and self.config.grad_norm_clip != 0.0:
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_norm_clip
            )
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _generic_loop(
        self,
        stage: Stage,
        loader: DataLoader,
        tracker: MetricsTracker,
        progress: Progress,
        max_epochs: int,
        iterating_cb: Callable[[int, int], bool],
        output_cb: Callable[[StepOutput], None] = None,
        task_name: str = None,
    ):
        loader_iter = iter(loader)

        self.progress.add_stage(stage, loader, task_name)
        self.trigger_callback(Callback.ON_EPOCH_START(stage))

        epoch = 0
        iterations = 0
        while iterating_cb(iterations, epoch):
            loader_iter, batch, epoch = self._get_batch(
                stage=stage,
                data_loader=loader,
                data_iter=loader_iter,
                progress=progress,
                epoch=epoch,
                max_epochs=max_epochs,
            )
            if batch is None:
                break

            self.trigger_callback(
                Callback.ON_ITER_START(stage), iterations=iterations, epoch=epoch
            )

            output = self.model.step(batch, stage)
            if output_cb:
                output_cb(output)

            # Compute batch and epoch metrics
            metrics = tracker.update(
                stage=stage, output=output, epoch=epoch, step=iterations
            )

            # Register metrics to wandb
            if self.config.wandb:
                self.wandb.log(stage=stage, metrics=metrics)

            # Advance the progress bar and log metrics
            self.progress.advance(stage=stage, metrics=metrics)

            self.trigger_callback(
                Callback.ON_ITER_END(stage),
                iterations=iterations,
                epoch=epoch,
                metrics=metrics,
            )

            iterations += 1

    @torch.no_grad()
    def _validation_loop(self, ev_loader):
        self.model.eval()

        iterating_cb = lambda i, _: i < len(ev_loader) or (
            self.config.eval_max_iterations is not None
            and i < self.config.eval_max_iterations
        )
        self._generic_loop(
            stage=Stage.VAL,
            loader=ev_loader,
            tracker=self.tracker,
            progress=self.progress,
            max_epochs=1,
            iterating_cb=iterating_cb,
        )

        self.progress.hide_stage(Stage.VAL)
        self.progress.set_stage(Stage.TRAIN)

        self.model.train()

    def _training_loop(self, tr_loader, ev_loader):
        def iterating_cb(i: int, e: int):
            cont = self.config.max_iterations is None or i < self.config.max_iterations
            if cont and self.config.evaluate and i % self.config.evaluate_every == 0:
                self._validation_loop(ev_loader)
            return cont

        def on_iter_start(_, **kwargs):
            self.iteration = kwargs["iterations"]
            self.epoch = kwargs["epoch"]

        id = self.add_callback(Callback.ON_ITER_START(Stage.TRAIN), on_iter_start)

        self.model.train()
        self.model.to(self.device)

        self._generic_loop(
            stage=Stage.TRAIN,
            loader=tr_loader,
            tracker=self.tracker,
            progress=self.progress,
            max_epochs=self.config.max_epochs,
            iterating_cb=iterating_cb,
            output_cb=self._fit_model,
            task_name="[blue]Epoch {epoch}[/blue]",
        )

        self.remove_callback(Callback.ON_ITER_START(Stage.TRAIN), id)

        self.model.cpu()
        self.model.eval()

    def fit(self, tr_dataset, ev_dataset=None):
        if self.config.evaluate and ev_dataset is None:
            log.warn(
                "You have chosen to evaluate the model, but no evaluation dataset is passed. Ignoring evaluation."
            )

        tr_loader = DataLoader(
            tr_dataset,
            sampler=RandomSampler(tr_dataset, replacement=True),
            shuffle=False,
            pin_memory=str(self.device) != "cpu",
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

        ev_loader = (
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

        self.date = get_date()

        if self.config.wandb:
            self.wandb.init(self.model, self.optimizer, self.scheduler)

        self.progress = Progress()
        with self.progress.bar:
            self._training_loop(tr_loader, ev_loader)

        self.save_model()

    def test(self, dataset):
        self.model.eval()

        progress = Progress()
        tracker = MetricsTracker()

        te_loader = DataLoader(
            dataset,
            shuffle=False,
            pin_memory=self.device != "cpu",
            batch_size=self.config.eval_batch_size,
            num_workers=self.config.eval_num_workers,
        )

        iterating_cb = lambda i, _: i < len(te_loader)

        with progress.bar:
            self._generic_loop(
                stage=Stage.TEST,
                loader=te_loader,
                tracker=tracker,
                progress=progress,
                max_epochs=1,
                iterating_cb=iterating_cb,
            )

        self.model.cpu()

        return tracker
