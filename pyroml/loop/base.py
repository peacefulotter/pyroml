import logging

from contextlib import nullcontext
from torch.utils.data import Dataset, DataLoader, RandomSampler

import pyroml as p
from pyroml.utils import Stage
from pyroml.callback import Callback
from pyroml.loop.status import Status
from pyroml.loop.autocast import Autocast
from pyroml.loop.progress_bar import ProgressBar
from pyroml.metrics.tracker import MetricsTracker

log = p.get_logger(__name__)


class Loop(Callback):
    def __init__(self, trainer: "p.Trainer", model: "p.PyroModel") -> None:
        self.trainer = trainer
        self.model = model

        self.status = Status(loop=self)
        self.progress = ProgressBar()
        self.tracker = MetricsTracker(loop=self)

        # Callbacks, in order of execution
        self.callbacks = [self]
        self.callbacks.extend(trainer.callbacks)
        self.callbacks.append(
            self.tracker
        )  # first to allow other callbacks to use metrics
        self.callbacks.append(self.model)

        self.temp_callbacks: list["p.Callback"] = []

        self.batch_size: int = None
        self.num_workers: int = None
        self.loader: DataLoader | None = None
        self.steps_per_epochs: int = None
        self.total_steps: int = None
        self._setup_done = False

    def _trigger_callback(
        self, hook_name: str, stage_callback: bool = True, **kwargs
    ) -> None:
        _hook_name = f"on_"
        if stage_callback:
            _hook_name += f"{self.stage.value}_"
        _hook_name += hook_name

        kwargs.update(self.status.to_dict())
        kwargs.update(dict(trainer=self.trainer, loop=self))

        for cb in self.callbacks + self.temp_callbacks:
            fn = getattr(cb, _hook_name)
            if not callable(fn):
                continue
            log.debug(f"Triggering callback {_hook_name} for {cb} with args {kwargs}")
            fn(**kwargs)

    @property
    def stage(self) -> "p.Stage":
        raise NotImplementedError

    @property
    def max_steps(self) -> int:
        raise NotImplementedError

    @property
    def max_epochs(self) -> int:
        raise NotImplementedError

    @property
    def autocast(self) -> Autocast:
        return self.trainer.autocast

    def before_step(self):
        pass

    def after_step(self, output: "p.StepOutput"):
        pass

    def _estimate_number_steps(self, loader: DataLoader):
        if self.max_epochs is None and self.max_steps is None:
            msg = "Either max_epochs or max_steps must be defined for training"
            raise ValueError(msg)

        if self.max_steps is not None and self.max_steps > 0:
            return self.max_steps

        return self.max_epochs * len(loader)

    def _setup(self, dataset: Dataset):
        if self._setup_done:
            return

        c = self.trainer
        is_training = self.stage == Stage.TRAIN

        self.batch_size = c.batch_size if is_training else c.eval_batch_size
        self.num_workers = c.num_workers if is_training else c.eval_num_workers
        sampler = RandomSampler(dataset, replacement=False) if is_training else None
        self.loader = DataLoader(
            dataset,
            sampler=sampler,
            shuffle=False,
            pin_memory=str(self.autocast.device) != "cpu",
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        self.steps_per_epochs: int = len(dataset) // self.batch_size
        self.total_steps: int = self._estimate_number_steps(dataset)
        self._setup_done = True

    def run(self, dataset: Dataset):
        self.temp_callbacks = [self.progress]
        self._setup(dataset)
        data_iter = iter(self.loader)

        self._trigger_callback("start")
        self._trigger_callback("epoch_start")

        if self.model.device != self.autocast.device:
            self.model = self.model.to(self.autocast.device)

        # progress bar context must not be opened twice (e.g. by train and val loop)
        with self.progress.bar if self.stage != Stage.VAL else nullcontext():
            while True:
                if self.status.step > self.total_steps:
                    break

                # --- Request next batch
                try:
                    batch = next(data_iter)
                except StopIteration:
                    self._trigger_callback("epoch_end")

                    if (
                        self.max_epochs is not None
                        and self.status.epoch + 1 >= self.max_epochs
                    ):
                        break

                    data_iter = iter(self.loader)
                    batch = next(data_iter)

                    self.status.advance_epoch()
                    self._trigger_callback("epoch_start")

                self.before_step()

                # --- Iteration starts
                self._trigger_callback("iter_start")

                # ----- Forward pass
                with self.autocast:
                    output = self.model.step(batch, self.stage)
                    self.after_step(output)

                # ----- Compute batch and epoch metrics
                self.tracker.update(output=output)

                self._trigger_callback("iter_end")
                self.status.advance_step()
                # --- Iteration ends

        self._trigger_callback("end")

        if self.stage != Stage.VAL:
            self.model = self.model.cpu()
