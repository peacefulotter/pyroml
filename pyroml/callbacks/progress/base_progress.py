from enum import Enum
from typing import TYPE_CHECKING

from typing_extensions import override

from pyroml import Stage
from pyroml.callbacks.callback import Callback
from pyroml.utils.log import get_logger

if TYPE_CHECKING:
    from pyroml.core.status import Status
    from pyroml.callbacks.callback import CallbackArgs
    from pyroml.loop.base import Loop

log = get_logger(__name__)


class ColorMode(Enum):
    TR = "blue"
    EV = "yellow"
    PR = "red"


class BaseProgress(Callback):
    def __init__(self, stack_bars: bool = True, rich_colors: bool = True):
        self.stack_bars = stack_bars
        self.rich_colors = rich_colors
        self._tasks: dict["Stage", int] = {}
        self._val_epoch_metrics: dict[str, float] = {}
        self._progress_stopped: bool = False

    @property
    def enabled(self):
        return not self._progress_stopped

    def setup():
        raise NotImplementedError()

    def add_task(self, status: "Status", total: int, desc: str = None) -> int:
        raise NotImplementedError()

    def reset_task(self, task: int, total: int, desc: str = None):
        raise NotImplementedError()

    def end_progress_task(self, task: int, hide: bool = False):
        raise NotImplementedError()

    def stop_progress(self):
        raise NotImplementedError()

    def update_metrics(self, task: int, metrics: str, advance: int = 1):
        raise NotImplementedError()

    def _setup(self):
        if self.enabled:
            return

        self.setup()
        self._tasks = {}
        self._val_epoch_metrics = {}
        self._progress_stopped = False

        log.debug("Setup done")

    def get_task(self, status: "Status"):
        return self._tasks.get(status.stage)

    def set_task(self, status: "Status", task: int):
        self._tasks[status.stage] = task

    # =================== internal api ===================

    def _add_stage(
        self,
        loop: "Loop",
        status: "Status",
        desc: str = None,
    ):
        assert self.enabled
        log.debug(f"Adding stage {status.stage.value}")

        total = len(loop.loader)
        task = self.get_task(status)

        if task is not None:
            self.reset_task(task, total, desc)
        else:
            task_id = self.add_task(status, total, desc)
            self.set_task(status, task_id)

    def _get_step_metrics(self, args: "CallbackArgs"):
        metrics = args.loop.tracker.get_last_step_metrics()
        if args.status.stage == Stage.TRAIN:
            metrics.update(self._val_epoch_metrics)
        return metrics

    def _get_epoch_metrics(self, args: "CallbackArgs"):
        metrics = args.loop.tracker.get_last_epoch_metrics()
        if args.status.stage == Stage.TRAIN:
            metrics.update(self._val_epoch_metrics)
        return metrics

    # =================== stop ===================

    def _stop_progress(self) -> None:
        if self.enabled:
            self.stop_progress()
            self._progress_stopped = True

    @override
    def on_exception(
        self,
        args: "CallbackArgs",
    ) -> None:
        self._stop_progress()

    # =================== start ===================

    def _on_start(self, args: "CallbackArgs"):
        self._setup()

    @override
    def on_train_start(self, args: "CallbackArgs"):
        self._on_start(args)

    @override
    def on_validation_start(self, args: "CallbackArgs"):
        self._on_start(args)
        self._val_epoch_metrics = {}

    @override
    def on_predict_start(self, args: "CallbackArgs"):
        self._on_start(args)

    # =================== epoch_start ===================

    def _on_epoch_start(self, desc: str, color: ColorMode, args: "CallbackArgs"):
        desc = f"[{color.value}]{desc}" if self.rich_colors else desc
        self._add_stage(status=args.status, loop=args.loop, desc=desc)

    @override
    def on_train_epoch_start(self, args: "CallbackArgs"):
        self._on_epoch_start(f"Epoch {args.status.epoch}", ColorMode.TR, args)

    @override
    def on_validation_epoch_start(self, args: "CallbackArgs"):
        self._on_epoch_start("Validating", ColorMode.EV, args)

    @override
    def on_predict_epoch_start(self, args: "CallbackArgs"):
        self._on_epoch_start("Predicting", ColorMode.PR, args)

    # =================== iter_end ===================

    def _on_iter(self, args: "CallbackArgs"):
        task = self.get_task(status=args.status)
        metrics = self._get_step_metrics(args=args)
        self.update_metrics(task=task, metrics=metrics, advance=1)

    @override
    def on_train_iter_end(self, args: "CallbackArgs"):
        self._on_iter(args)

    @override
    def on_validation_iter_end(self, args: "CallbackArgs"):
        self._on_iter(args)

    @override
    def on_predict_iter_end(self, args: "CallbackArgs"):
        self._on_iter(args)

    # =================== epoch_end ===================

    def _on_epoch_end(
        self,
        args: "CallbackArgs",
        hide: bool = False,
    ):
        task = self.get_task(status=args.status)
        if not hide:
            metrics = self._get_epoch_metrics(args=args)
            self.update_metrics(task=task, metrics=metrics, advance=0)
        self.end_progress_task(task, hide)
        self.set_task(status=args.status, task=None)

    @override
    def on_train_epoch_end(self, args: "CallbackArgs"):
        self._on_epoch_end(args)

    @override
    def on_validation_epoch_end(self, args: "CallbackArgs"):
        # # If a training task exists, move validation metrics to training progress bar
        # # TODO: this won't work for validation sanity check, add a way to make it work (add Stage.SANITY ?)
        # tr_task = self._tasks.get(Stage.TRAIN)
        # if tr_task is not None:

        # Save validation epoch metrics to use for the training progress bar
        metrics = args.loop.tracker.get_last_epoch_metrics()
        metrics = {
            f"{args.status.stage.to_prefix()}_{k}": v for k, v in metrics.items()
        }
        self._val_epoch_metrics = metrics

        self._on_epoch_end(args, hide=True)

    @override
    def on_predict_epoch_end(self, args: "CallbackArgs"):
        self._on_epoch_end(args)
