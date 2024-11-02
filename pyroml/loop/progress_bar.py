from enum import Enum
from rich.progress import (
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
    Progress,
)


import pyroml as p
from pyroml.callback import Callback
from pyroml.utils import Stage, Singleton
from pyroml.metrics.tracker import EPOCH_PREFIX


class ColorMode(Enum):
    TR = "blue"
    EV = "yellow"
    TE = "red"
    PR = "red"


class SingletonBar(Progress, metaclass=Singleton):
    def __init__(self):
        super().__init__(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("/"),
            TimeRemainingColumn(),
            TextColumn("•"),
            TextColumn("{task.fields[metrics]}"),
        )
        self.registered_tasks = {}


class ProgressBar(Callback):
    def __init__(self):
        self.bar = SingletonBar()
        self.metrics: dict[str, float] = {}

    def _get_task(self, loop: "p.Loop"):
        return self.bar.registered_tasks.get(loop.status.stage)

    def _set_task(self, loop: "p.Loop", task):
        self.bar.registered_tasks[loop.status.stage] = task

    def _on_start(self, desc: str, color: ColorMode, **kwargs: "p.CallbackKwargs"):
        loop = kwargs["loop"]
        desc = f"[{color.value}]{desc}"
        self._add_stage(loop=loop, desc=desc)

    def _on_iter(self, **kwargs: "p.CallbackKwargs"):
        loop = kwargs["loop"]
        self._advance(loop)

    def _on_end(self, **kwargs: "p.CallbackKwargs"):
        task = self._get_task(kwargs["loop"])
        if task is None:
            return
        self.bar.update(task, visible=False)
        self.bar.stop_task(task)
        self.bar.remove_task(task)
        self._set_task(kwargs["loop"], None)

    def on_train_epoch_start(self, **kwargs: "p.CallbackKwargs"):
        self._on_start(f"Epoch {kwargs['loop'].status.epoch}", ColorMode.TR, **kwargs)

    def on_validation_epoch_start(self, **kwargs: "p.CallbackKwargs"):
        self._on_start("Validating", ColorMode.EV, **kwargs)

    def on_test_epoch_start(self, **kwargs: "p.CallbackKwargs"):
        self._on_start("Testing", ColorMode.TE, **kwargs)

    def on_train_epoch_end(self, **kwargs: "p.CallbackKwargs"):
        task = self._get_task(kwargs["loop"])
        self.bar.stop_task(task)
        self._set_task(kwargs["loop"], None)

    def on_train_iter_end(self, **kwargs: "p.MetricsKwargs"):
        self._on_iter(**kwargs)

    def on_validation_iter_end(self, **kwargs: "p.MetricsKwargs"):
        self._on_iter(**kwargs)

    def on_test_iter_end(self, **kwargs: "p.MetricsKwargs"):
        self._on_iter(**kwargs)

    def on_train_end(self, **kwargs: "p.CallbackKwargs"):
        self._on_end(**kwargs)

    def on_validation_end(self, **kwargs: "p.CallbackKwargs"):
        self._on_end(**kwargs)

    def on_test_end(self, **kwargs: "p.CallbackKwargs"):
        self._on_end(**kwargs)

    def _add_stage(
        self,
        loop: "p.Loop",
        desc: str = None,
    ):
        total = len(loop.loader)
        task = self._get_task(loop)

        if task is not None:
            self.bar.reset(
                task,
                total=total,
                advance=0,
                visible=False,
                metrics="",
                description=desc,
            )
        else:
            task = self.bar.add_task(
                metrics="",
                total=total,
                description=desc,
            )
            self._set_task(loop, task)

    def _prefix(self, stage: "p.Stage", name: str):
        if stage == Stage.TRAIN:
            return name
        return f"{stage.to_prefix()}_{name}"

    def _register_metrics(self, metrics: dict[str, float], stage: "p.Stage"):
        """Register the metrics to be displayed in the progress bar"""
        for name, value in metrics.items():
            name = self._prefix(stage, name)
            self.metrics[name] = value

    def metrics_to_str(self):
        metrics_str = ""
        for name, value in self.metrics.items():
            metrics_str += f"{name}={value:.3f} "
        return metrics_str

    def update_metrics(
        self, loop: "p.Loop", metrics: dict[str, float], stage: "p.Stage", advance=1
    ):
        task = self._get_task(loop)
        self._register_metrics(metrics, stage)
        metrics_str = self.metrics_to_str()
        self.bar.update(task, metrics=metrics_str, advance=advance)
        self.bar.refresh()

    def replace_metrics_with_epoch_metrics(self, metrics, epoch_metrics):
        for name, value in epoch_metrics.items():
            real_name = name[len(EPOCH_PREFIX) + 1 :]
            metrics[real_name] = value
        return metrics

    def _advance(
        self,
        loop: "p.Loop",
    ):
        metrics = loop.tracker.get_last_step_metrics()
        epoch_metrics = loop.tracker.get_last_epoch_metrics()
        metrics = self.replace_metrics_with_epoch_metrics(metrics, epoch_metrics)
        self.update_metrics(loop, metrics, stage=loop.status.stage)
