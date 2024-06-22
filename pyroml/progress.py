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
        self.tasks: object[Stage, TaskID] = {}
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

    def _unroll_metrics(self, metrics, name=None):
        if isinstance(metrics, dict):
            name = "" if name is None else name + "_"
            metrics = [
                self._unroll_metrics(v, name=f"{name}{k}") for k, v in metrics.items()
            ]
            return " ".join(metrics)
        name = name or "loss"
        return f"{name}={metrics:.3f}"

    def advance(self, metrics={}):
        kwargs = {}
        kwargs["metrics"] = self._unroll_metrics(metrics)
        kwargs["advance"] = 1
        self.bar.update(self.current_task, **kwargs)

    def stop(self):
        self.bar.console.print("stopping " + str(self.current_task))
