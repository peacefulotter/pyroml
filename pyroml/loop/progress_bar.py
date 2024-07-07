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


class ProgressBar(Callback):
    def __init__(self):
        self.bar = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("/"),
            TimeRemainingColumn(),
            TextColumn("•"),
            TextColumn("{task.fields[metrics]}"),
        )

        self.tasks: dict[Stage, TaskID] = {}
        self.metrics: dict[str, float] = {}

    def on_train_epoch_start(self, trainer: "p.Trainer", **kwargs: "p.CallbackKwargs"):
        length = len(trainer.train_loader)
        self._add_stage(
            stage=Stage.TRAIN, length=length, description="[blue]Epoch {epoch}[/blue]"
        )

    def on_validation_epoch_start(
        self, trainer: "p.Trainer", **kwargs: "p.CallbackKwargs"
    ):
        length = len(trainer.val_loader)
        self._add_stage(stage=Stage.VAL, length=length, description="Validating")

    def on_test_epoch_start(self, trainer: "p.Trainer", **kwargs: "p.CallbackKwargs"):
        length = len(trainer.test_loader)
        self._add_stage(stage=Stage.TEST, length=length, description="Testing")

    def on_train_iter_end(self, trainer: "p.Trainer", **kwargs: "p.MetricsKwargs"):
        self._advance(stage=Stage.TRAIN, **kwargs)

    def on_validation_iter_end(self, trainer: "p.Trainer", **kwargs: "p.MetricsKwargs"):
        self._advance(stage=Stage.VAL, **kwargs)

    def on_test_iter_end(self, trainer: "p.Trainer", **kwargs: "p.MetricsKwargs"):
        self._advance(stage=Stage.TEST, **kwargs)

    def on_validation_end(self, trainer: "p.Trainer", **kwargs: "p.CallbackKwargs"):
        if Stage.VAL in self.tasks:
            self.bar.remove_task(self.tasks[Stage.VAL])
            del self.tasks[Stage.VAL]

    def _add_stage(
        self,
        stage: "p.Stage",
        length: int,
        description: str = None,
    ):
        task = self.bar.add_task(
            description=description,
            total=length,
            metrics="",
        )
        self.tasks[stage] = task

    def _prefix(self, stage: "p.Stage", name: str):
        if stage == Stage.TRAIN:
            return name
        return f"{stage.to_prefix()}_{name}"

    def _register_metrics(self, stage: "p.Stage", metrics: dict[str, float]):
        """Register the metrics to be displayed in the progress bar and convert them to string."""
        str_metrics = ""

        for name, value in metrics.items():
            name = self._prefix(stage, name)
            self.metrics[name] = value
            str_metrics += f"{name}={value:.3f} "

        return str_metrics

    def _advance(self, stage: "p.Stage", **kwargs: "p.MetricsKwargs"):
        epoch = kwargs["epoch"]
        metrics = kwargs["metrics"]

        metrics_str = self._register_metrics(stage, metrics)

        current_task = self.tasks[stage]
        kwargs = dict(
            epoch=epoch,
            metrics=metrics_str,
            advance=1,
        )
        self.bar.update(current_task, **kwargs)
