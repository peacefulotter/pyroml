from rich.progress import (
    TaskID,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
    Progress,
)


from pyroml.utils import Stage
from pyroml.status import Status
from pyroml.callback import Callback


class ProgressBar(Callback):
    def __init__(self, status: Status):
        self.status = status

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

    @property
    def _current_task(self) -> TaskID:
        if self.status.stage not in self.tasks:
            raise ValueError("No task found for the current stage")
        return self.tasks[self.status.stage]

    def add_stage(
        self,
        length: int,
        task_name: str = None,
    ):
        stage = self.status.stage
        desc = task_name or stage.to_progress()
        task = self.bar.add_task(
            description=desc,
            total=length,
            metrics="",
        )
        self.tasks[stage] = task

    def on_stage_change(self, **kwargs):
        old_stage = kwargs["old_stage"]
        if (
            old_stage is not None
            and old_stage != Stage.TRAIN
            and old_stage in self.tasks
        ):
            self.bar.remove_task(self.tasks[old_stage])
            del self.tasks[old_stage]

    def _prefix(self, name: str):
        stage = self.status.stage
        if stage == Stage.TRAIN:
            return name
        return f"{stage.to_prefix()}_{name}"

    def _register_metrics(self, metrics: dict[str, float]):
        for name, value in metrics.items():
            name = self._prefix(name)
            self.metrics[name] = value

    def _metrics_to_str(self):
        str_metrics = ""
        for name, value in self.metrics.items():
            str_metrics += f"{name}={value:.3f} "
        return str_metrics

    def advance(self, metrics: dict[str, float] = {}):
        # TODO: log metrics to a txt file too ? do that here or dedicated file logger rather?

        self._register_metrics(metrics)
        metrics_str = self._metrics_to_str()

        kwargs = dict(
            metrics=metrics_str,
            advance=1,
        )
        self.bar.update(self._current_task, **kwargs)
