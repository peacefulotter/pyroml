import logging

from torch.utils.data import Dataset, DataLoader, RandomSampler

import pyroml as p
from pyroml.utils import Stage
from pyroml.loop.status import Status
from pyroml.loop.autocast import Autocast
from pyroml.loop.progress_bar import ProgressBar
from pyroml.metrics.tracker import MetricsTracker

log = logging.getLogger(__name__)


class Loop:
    def __init__(self, trainer: "p.Trainer", model: "p.PyroModel"):
        self.trainer = trainer
        self.model = model

        # Context manager for mixed precision training and moving to device - On cpu, only bfloat16 is supported
        self.autocast = Autocast(trainer)

        self.status = Status(loop=self)
        self.progress = ProgressBar(loop=self)
        self.tracker = MetricsTracker(loop=self)

        # Callbacks, in order of execution
        self.callbacks = trainer.callbacks
        self.callbacks.append(self.model)
        # self.callbacks.append(self.autocast)

        self.temp_callbacks: list["p.Callback"] = []

        self.loader: DataLoader | None = None

    def _trigger_callback(self, hook_name: str, stage_callback: bool = True, **kwargs):
        _hook_name = f"on_"
        if stage_callback:
            _hook_name += f"{self.stage.value}_"
        _hook_name += hook_name

        kwargs.update(self.status.to_dict())

        for cb in self.callbacks + self.temp_callbacks:
            fn = getattr(cb, _hook_name)
            if not callable(fn):
                continue
            log.debug(f"Triggering callback {_hook_name} for {cb} with args {kwargs}")
            fn(self.trainer, self, **kwargs)

    @property
    def stage(self):
        raise NotImplementedError

    @property
    def max_steps(self):
        raise NotImplementedError

    @property
    def max_epochs(self):
        raise NotImplementedError

    def before_step(self):
        pass

    def after_step(self, output: "p.StepOutput"):
        pass

    def _get_dataloader(self, dataset: Dataset):
        c = self.trainer
        is_training = self.stage == Stage.TRAIN
        batch_size = c.batch_size if is_training else c.eval_batch_size
        num_workers = c.num_workers if is_training else c.eval_num_workers
        sampler = RandomSampler(dataset, replacement=True) if is_training else None

        return DataLoader(
            dataset,
            sampler=sampler,
            shuffle=False,
            pin_memory=str(self.autocast.device) != "cpu",
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def run(self, dataset: Dataset):
        self.temp_callbacks = [self.progress]

        self._trigger_callback("start")

        self.loader = self._get_dataloader(dataset)
        data_iter = iter(self.loader)

        if self.max_epochs is None and self.max_steps is None:
            msg = "Either max_epochs or max_steps must be defined for training"
            raise ValueError(msg)

        self._trigger_callback("epoch_start")

        if self.model.device != self.autocast.device:
            self.model = self.model.to(self.autocast.device)

        while True:
            if self.max_steps and self.status.step >= self.max_steps:
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
            metrics = self.tracker.update(output=output)

            self._trigger_callback("iter_end", metrics=metrics)
            self.status.advance_step()
            # --- Iteration ends

        self._trigger_callback("end")

        if self.stage != Stage.VAL:
            self.model = self.model.cpu()
