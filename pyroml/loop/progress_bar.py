from rich.progress import (
    TaskID,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
    Progress,
)


import pyroml as p
from pyroml.utils import Stage
from pyroml.callback import Callback


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


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


class ProgressBar(Callback):
    def __init__(self, loop: "p.Loop"):
        self.status = loop.status
        self.bar = SingletonBar()

        self.task = None
        self.metrics: dict[str, float] = {}

    def on_train_epoch_start(self, **kwargs: "p.CallbackKwargs"):
        loop = kwargs["loop"]
        self._add_stage(loop=loop, desc=f"[blue]Epoch {loop.status.epoch}[/blue]")

    def on_train_epoch_end(self, **kwargs: "p.CallbackKwargs"):
        self.bar.stop_task(self.task)
        self.task = None

    def on_validation_epoch_start(self, **kwargs: "p.CallbackKwargs"):
        loop = kwargs["loop"]
        self._add_stage(loop=loop, desc="Validating")

    def on_test_epoch_start(self, **kwargs: "p.CallbackKwargs"):
        loop = kwargs["loop"]
        self._add_stage(loop=loop, desc="Testing")

    def on_train_iter_end(self, **kwargs: "p.MetricsKwargs"):
        loop = kwargs["loop"]
        self._advance(loop)

    def on_validation_iter_end(self, **kwargs: "p.MetricsKwargs"):
        loop = kwargs["loop"]
        self._advance(loop)

    def on_validation_end(self, **kwargs: "p.CallbackKwargs"):
        self.bar.update(self.task, visible=False)
        self.bar.stop_task(self.task)
        self.bar.remove_task(self.task)
        self.task = None

    def on_test_iter_end(self, **kwargs: "p.MetricsKwargs"):
        loop = kwargs["loop"]
        self._advance(loop)

    def on_test_end(self, **kwargs: "p.CallbackKwargs"):
        self.bar.stop_task(self.task)
        self.task = None

    def _add_stage(
        self,
        loop: "p.Loop",
        desc: str = None,
    ):
        total = len(loop.loader)
        self.task = self.bar.add_task(
            metrics="",
            total=total,
            description=desc,
        )

    def _prefix(self, stage: "p.Stage", name: str):
        if stage == Stage.TRAIN:
            return name
        return f"{stage.to_prefix()}_{name}"

    def _register_metrics(self, metrics: dict[str, float]):
        """Register the metrics to be displayed in the progress bar and convert them to string."""
        stage = self.status.stage
        str_metrics = ""

        for name, value in metrics.items():
            name = self._prefix(stage, name)
            self.metrics[name] = value
            str_metrics += f"{name}={value:.3f} "

        return str_metrics

    def _advance(
        self,
        loop: "p.Loop",
        desc: str = None,
    ):
        metrics = loop.tracker.get_last_step_metrics()
        metrics.update(loop.tracker.get_last_epoch_metrics())
        metrics_str = self._register_metrics(metrics)

        kwargs = dict(
            metrics=metrics_str,
            advance=1,
        )
        if desc is not None:
            kwargs["description"] = desc
        self.bar.update(self.task, **kwargs)
