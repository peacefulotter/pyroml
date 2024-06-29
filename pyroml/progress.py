from torch.utils.data import DataLoader
from rich.progress import (
    TaskID,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
    Progress as ProgressBar,
)


from pyroml.utils import Stage


class Progress:
    def __init__(self):
        self.bar = ProgressBar(
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
        self.current_task: TaskID = None

    def add_stage(
        self,
        stage: Stage,
        loader: DataLoader,
        task_name: str = None,
    ):
        desc = task_name or stage.to_progress()
        task = self.bar.add_task(
            description=desc,
            total=len(loader),
            metrics="",
        )
        self.tasks[stage] = task
        self.current_task = task

    def hide_stage(self, stage: Stage):
        if stage in self.tasks:
            self.bar.remove_task(self.tasks[stage])
            del self.tasks[stage]

    def set_stage(self, stage: Stage):
        if stage in self.tasks:
            self.current_task = self.tasks[stage]

    def _prefix(self, stage: Stage, name: str):
        if stage == Stage.TRAIN:
            return name
        return f"{stage.to_prefix()}_{name}"

    def _register_metrics(self, stage: Stage, metrics: dict[str, float]):
        for name, value in metrics.items():
            name = self._prefix(stage, name)
            self.metrics[name] = value

    def _metrics_to_str(self):
        str_metrics = ""
        for name, value in self.metrics.items():
            str_metrics += f"{name}={value:.3f} "
        return str_metrics

    def advance(self, stage: Stage, metrics: dict[str, float] = {}):
        self._register_metrics(stage, metrics)
        metrics_str = self._metrics_to_str()

        kwargs = {}
        kwargs["metrics"] = metrics_str
        kwargs["advance"] = 1
        self.bar.update(self.current_task, **kwargs)

    def stop(self):
        self.bar.console.print("stopping " + str(self.current_task))
