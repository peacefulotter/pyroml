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
        self.tr_task: TaskID = None
        self.ev_task: TaskID = None
        self.current_task: TaskID = None

    def new_epoch(self, epoch):
        self.tr_task = self.bar.add_task(
            f"Epoch {epoch}" + f" - {Stage.TRAIN.to_progress()}", metrics=""
        )
        self.current_task = self.tr_task

    def set_stage(self, stage: Stage, loader: DataLoader = None):
        if stage == Stage.TRAIN:
            if self.ev_task is None:
                self.bar.update(self.tr_task, total=len(loader))
            else:
                self.bar.update(self.ev_task, visible=False)
                self.ev_task = None
            self.current_task = self.tr_task
        elif stage == Stage.VAL:
            self.ev_task = self.bar.add_task(
                description=stage.to_progress(), total=len(loader), metrics=""
            )
            self.current_task = self.ev_task

    def _parse_metrics(self, metrics):
        return " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])

    def advance(self, metrics={}):
        kwargs = {}
        kwargs["metrics"] = self._parse_metrics(metrics)
        kwargs["advance"] = 1
        self.bar.update(self.current_task, **kwargs)

    def stop(self):
        self.bar.console.print("stopping " + str(self.current_task))
