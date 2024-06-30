import torch
import logging

from typing import Callable
from contextlib import nullcontext
from torch.utils.data import Dataset


from pyroml.autocast import Autocast
from pyroml.callback import Callback
from pyroml.status import Status
from pyroml.utils import Stage
from pyroml.config import Config
from pyroml.progress_bar import ProgressBar
from pyroml.wandb_logger import Wandb
from pyroml.batch_iterator import BatchIterator
from pyroml.checkpoint import Checkpoint
from pyroml.metrics_tracker import MetricsTracker
from pyroml.model import PyroModel, StepOutput

log = logging.getLogger(__name__)


class Trainer:
    # TODO: merge config into trainer? in that case, model should be passed as parameter to all public methods
    def __init__(self, model: PyroModel, config: Config):
        self.model = model
        self.config = config

        if self.config.verbose:
            log.setLevel(logging.INFO)
        if self.config.debug:
            log.setLevel(logging.DEBUG)

        # Context manager for mixed precision training and moving to device - On cpu, only bfloat16 is supported
        self.autocast = Autocast(config)

        # Compile model if requested, improves performance
        if self.config.compile:
            self.compile_model()

        self.model.__setattr__("_trainer", self)
        print(self.model._trainer)

        self.status = Status(self)
        self.metrics_tracker = MetricsTracker(self.model, self.status)

        self.callbacks = config.callbacks
        self.callbacks.append(self.model)
        self.callbacks.append(self.autocast)

        if config.wandb:
            wandb = Wandb(model=self.model, config=config)
            self.callbacks.append(wandb)

        self.temp_callbacks: list[Callback] = []

        self.progress: ProgressBar = None

    def _trigger_callback(self, hook_name: str, stage_callback: bool = True, **kwargs):
        _hook_name = f"on_"
        if stage_callback:
            _hook_name += f"{self.status.stage.value}_"
        _hook_name += hook_name

        kwargs["trainer"] = self
        kwargs["epoch"] = self.status.epoch
        kwargs["step"] = self.status.step

        for cb in self.callbacks + self.temp_callbacks:
            fn = getattr(cb, _hook_name)
            if not callable(fn):
                continue
            log.debug(f"Triggering callback {_hook_name} for {cb} with args {kwargs}")
            fn(**kwargs)

    def compile_model(self):
        if self.model._is_compiled():
            log.info("Model is already compiled, skipping compilation")
            return

        log.info(f"Compiling model...")
        self.model = torch.compile(self.model)
        log.info(f"Model compiled!")

    def _extract_loss(self, output):
        if isinstance(output, dict):
            return self._extract_loss(output["loss"])
        elif isinstance(output, torch.Tensor):
            return output
        else:
            raise ValueError(
                "The return type of the model.step function should be a torch.Tensor, or dict[torch.Tensor]"
            )

    def _generic_loop(
        self,
        stage: Stage,
        dataset: Dataset,
        progress: ProgressBar,
        before_forward_cb: Callable = None,
        after_forward_cb: Callable[[StepOutput], None] = None,
        task_name: str = None,
    ):
        self.temp_callbacks = [progress]
        old_stage = self.status.stage
        self.status.stage = stage
        self._trigger_callback("start")

        def iterate(batch):
            # Forward pass
            with self.autocast:
                if before_forward_cb:
                    before_forward_cb()

                output = self.model.step(batch, stage)

                if after_forward_cb:
                    after_forward_cb(output)

            # Compute batch and epoch metrics
            metrics = self.metrics_tracker.update(output=output)

            # Advance the progress bar and log metrics
            progress.advance(metrics=metrics)

            return metrics

        BatchIterator.iterate(
            trainer=self,
            dataset=dataset,
            progress=progress,
            task_name=task_name,
            cb=iterate,
        )

        self._trigger_callback("end")
        self.status.stage = old_stage
        self.temp_callbacks = []

    @torch.no_grad()
    def _validation_loop(self, dataset):
        self._generic_loop(
            stage=Stage.VAL,
            dataset=dataset,
            progress=self.progress,
        )

    # TODO: Move this to a dedicated Loop(Callback) class
    # such that we can remove the before_forward_cb and after_forward_cb
    def _training_loop(self, tr_dataset: Dataset, ev_dataset: Dataset):

        print(self.model._trainer)
        self.model.configure_optimizers()

        def before_forward_cb():
            if (
                self.config.evaluate
                and self.status.step % self.config.evaluate_every == 0
            ):
                self._validation_loop(ev_dataset)

        self._generic_loop(
            stage=Stage.TRAIN,
            dataset=tr_dataset,
            progress=self.progress,
            before_forward_cb=before_forward_cb,
            after_forward_cb=self.model._fit,
            task_name="[blue]Epoch {epoch}[/blue]",
        )

    def fit(self, tr_dataset: Dataset, ev_dataset: Dataset = None):
        if self.config.evaluate and ev_dataset is None:
            log.warn(
                "You have chosen to evaluate the model, but no evaluation dataset is passed. Ignoring evaluation."
            )

        self.progress = ProgressBar(self.status)
        with self.progress.bar:
            self._training_loop(tr_dataset, ev_dataset)

        return self.metrics_tracker

    def test(self, dataset):
        progress = ProgressBar(self.status)
        with progress.bar:
            self._generic_loop(
                stage=Stage.TEST,
                dataset=dataset,
                progress=progress,
            )

        return self.metrics_tracker

    # @staticmethod
    # def load(
    #     folder: Path | str = None,
    #     resume: bool = True,
    #     strict: bool = True,
    # ):
    #     """
    #     Loads a pretrained model from a specified checkpoint folder.

    #     Args:
    #         folder (str): The folder path where the pretrained model is saved.
    #         resume (bool, optional): Whether to resume training from the checkpoint. Defaults to True.
    #         strict (bool, optional): Whether to strictly enforce the shape and type of the loaded weights. Defaults to True.
    #     """
