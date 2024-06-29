import torch
import logging

from typing import Callable
from contextlib import nullcontext
from torch.utils.data import Dataset, DataLoader


from pyroml.config import Config
from pyroml.progress import Progress
from pyroml.wandb_logger import Wandb
from pyroml.dispenser import Dispenser
from pyroml.checkpoint import Checkpoint
from pyroml.tracker import MetricsTracker
from pyroml.model import PyroModel, StepOutput
from pyroml.utils import Callback, CallbackHandler, Stage

log = logging.getLogger(__name__)


class Trainer(CallbackHandler):
    # TODO: merge config into trainer? in that case, model should be passed as parameter to all public methods
    def __init__(self, model: PyroModel, config: Config):
        CallbackHandler.__init__(self)

        self.model = model
        self.config = config

        if self.config.verbose:
            log.setLevel(logging.INFO)
        if self.config.debug:
            log.setLevel(logging.DEBUG)

        # Device selection, auto will default to cuda if available
        device_type = config.device
        if device_type == "auto":
            device_type = "cuda" if torch.cuda.is_available() else "cpu"

        # Context manager for mixed precision training - On cpu, only bfloat16 is supported
        self.type_ctx = (
            nullcontext()
            if device_type == "cpu" and config.dtype != torch.bfloat16
            else torch.autocast(device_type=device_type, dtype=config.dtype)
        )
        log.info(
            f"Using device {self.type_ctx.device}, dtype {self.type_ctx.fast_dtype}"
        )

        # Compile model if requested, improves performance
        if self.config.compile:
            self.compile_model()

        if config.wandb:
            self.wandb = Wandb(self, config)

        self.tracker = MetricsTracker(self.model)
        self.progress: Progress = None

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
        progress: Progress,
        output_cb: Callable[[StepOutput], None] = None,
        task_name: str = None,
    ):
        self.trigger_callback(Callback.ON_START(stage))

        dispenser = Dispenser(self.config, dataset, stage)

        progress.add_stage(stage, loader, task_name)

        # TODO: Merge some callbacks, iterating_cb, and _get_batch. Make it return an iterator here
        while iterating_cb(step, epoch):

            self.trigger_callback(Callback.ON_ITER_START(stage), step=step, epoch=epoch)

            with self.type_ctx:
                output = self.model.step(batch, stage)
                if output_cb:
                    output_cb(output)

            # Compute batch and epoch metrics
            metrics = self.tracker.update(
                stage=stage, output=output, epoch=epoch, step=step
            )

            # Advance the progress bar and log metrics
            progress.advance(stage=stage, metrics=metrics)

            self.trigger_callback(
                Callback.ON_ITER_END(stage),
                step=step,
                epoch=epoch,
                metrics=metrics,
            )

            step += 1

        self.trigger_callback(Callback.ON_END(stage))

    @torch.no_grad()
    def _validation_loop(self, ev_loader):
        self.model.eval()

        self._generic_loop(
            stage=Stage.VAL,
            loader=ev_loader,
            progress=self.progress,
        )

        self.progress.hide_stage(Stage.VAL)
        self.progress.set_stage(Stage.TRAIN)

        self.model.train()

    def _training_loop(self, tr_dataset: Dataset, ev_dataset: Dataset):

        # TODO: Move this to a dedicated class, handling batch iteration and iter / epoch callbacks
        def iterating_cb(i: int, e: int):
            cont = self.config.max_steps is None or i < self.config.max_steps
            if cont and self.config.evaluate and i % self.config.evaluate_every == 0:
                self._validation_loop(ev_dataset)
            return cont

        self.model.train()
        self.model.to(self.type_ctx.device)

        self._generic_loop(
            stage=Stage.TRAIN,
            dataset=tr_dataset,
            progress=self.progress,
            output_cb=self.model._fit,
            task_name="[blue]Epoch {epoch}[/blue]",
        )

        self.model.cpu()
        self.model.eval()

    def fit(self, tr_dataset: Dataset, ev_dataset: Dataset = None):
        if self.config.evaluate and ev_dataset is None:
            log.warn(
                "You have chosen to evaluate the model, but no evaluation dataset is passed. Ignoring evaluation."
            )

        self.progress = Progress()
        with self.progress.bar:
            self._training_loop(tr_dataset, ev_dataset)

        return self.tracker

    def test(self, dataset):
        self.model.eval()
        self.model.to(self.type_ctx.device)

        progress = Progress()
        with progress.bar:
            self._generic_loop(
                stage=Stage.TEST,
                dataset=dataset,
                progress=progress,
            )

        self.model.cpu()
        return self.tracker

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
