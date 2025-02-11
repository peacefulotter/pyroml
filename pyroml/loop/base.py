from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from pyroml.callbacks import Callback, CallbackArgs
from pyroml.core.stage import Stage
from pyroml.core.status import Status
from pyroml.core.tracker import MetricsTracker
from pyroml.utils.device import to_device
from pyroml.utils.log import get_logger

if TYPE_CHECKING:
    from pyroml.callbacks import Callback
    from pyroml.core.autocast import Autocast
    from pyroml.core.model import PyroModel
    from pyroml.core.trainer import Trainer

log = get_logger(__name__)


class Loop:
    def __init__(
        self, trainer: "Trainer", model: "PyroModel", dataset: Dataset
    ) -> None:
        self.trainer = trainer
        self.model = model

        self.status = Status(stage=self.stage)
        self.tracker = MetricsTracker(status=self.status)

        # Callbacks, in order of execution
        # Tracker is first to expose metrics to other callbacks
        self.callbacks: list[Callback] = [
            self.tracker,
            self.model,
            *trainer.callbacks,
        ]

        is_training = self.stage == Stage.TRAIN
        sampler = RandomSampler(dataset, replacement=False) if is_training else None
        pin_memory = (
            str(self.device) != "cpu"
            if trainer.pin_memory is None
            else trainer.pin_memory
        )

        self.loader = DataLoader(
            dataset,
            sampler=sampler,
            shuffle=False,
            pin_memory=pin_memory,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        self.steps_per_epoch: int = len(self.loader)
        self.total_steps: int = self._estimate_number_steps(dataset)

    @property
    def stage(self) -> "Stage":
        raise NotImplementedError

    @property
    def max_steps(self) -> int:
        raise NotImplementedError

    @property
    def max_epochs(self) -> int:
        raise NotImplementedError

    @property
    def batch_size(self) -> int:
        raise NotImplementedError

    @property
    def num_workers(self) -> int:
        raise NotImplementedError

    @property
    def autocast(self) -> "Autocast":
        return self.trainer.autocast

    @property
    def device(self) -> torch.device:
        return self.autocast.device

    @property
    def dataset(self) -> Dataset:
        return self.loader.dataset

    def log(self, **data: dict[str, float | np.ndarray | torch.Tensor]):
        return self.tracker.log(**data)

    def before_step(self):
        pass

    def after_step(self, output: torch.Tensor | Any):
        pass

    def after_epoch(self):
        pass

    def _get_callback_args(self):
        return CallbackArgs(
            trainer=self.trainer,
            loop=self,
            model=self.model,
            status=self.status,
        )

    def _trigger_callback(
        self, hook_name: str, stage_independent: bool = False
    ) -> None:
        args = self._get_callback_args()
        hook_name = (
            f"on{'' if stage_independent else '_' + self.stage.value}_{hook_name}"
        )
        for cb in self.callbacks:
            cb: "Callback"
            log.debug(f"Triggering callback {hook_name} for {cb} with args {args}")
            cb._call_event(hook_name, args)

    def _estimate_number_steps(self, loader: DataLoader):
        if self.max_epochs is None and self.max_steps is None:
            msg = "Either max_epochs or max_steps must be defined for a loop to run"
            raise ValueError(msg)

        if self.max_steps is not None and self.max_steps > 0:
            return self.max_steps

        return self.max_epochs * len(loader)

    def _run(self):
        data_iter = iter(self.loader)

        self._trigger_callback("start")
        self._trigger_callback("epoch_start")

        if self.model.device != self.device:
            self.model.to(self.device)

        while True:
            if self.status.step > self.total_steps:
                break

            # --- Request next batch
            try:
                batch = next(data_iter)

            except StopIteration:
                # --- Epoch ends
                self._trigger_callback("epoch_end")
                self.after_epoch()

                if self.max_epochs is not None and self.status.epoch >= self.max_epochs:
                    break

                data_iter = iter(self.loader)
                batch = next(data_iter)

                # --- Epoch starts
                self.status.advance_epoch()
                self._trigger_callback("epoch_start")

            self.before_step()

            # --- Iteration starts
            self._trigger_callback("iter_start")

            # ----- Forward pass
            batch = to_device(batch, self.device, non_blocking=True)
            with self.autocast:
                loss = self.model.step(batch, self.stage)

            self.after_step(loss)

            # --- Iteration ends
            self._trigger_callback("iter_end")
            self.status.advance_step()

        self._trigger_callback("end")

        # Move model back to CPU only if no other loop is queued
        if self.trainer.status_stack_empty:
            self.model.cpu()

        return self.tracker

    def run(self):
        try:
            return self._run()
        except Exception as e:
            self._trigger_callback("exception", stage_independent=True)
            raise e
