from rich.progress import (
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
    Progress as ProgressBar,
)


from pyroml.utils import Stage


class Progress(ProgressBar):
    def __init__(self):
        super().__init__(
            TextColumn("\t[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        )
        self.tr_task = self.bar.add_task("Epoch 0", total=len(self.train_loader))
        self.ev_task = self.bar.add_task("Evaluating", total=len(self.train_loader))

    def _get_task_from_stage(self, stage: Stage):
        if stage == Stage.TRAIN:
            return self.tr_task
        elif stage == Stage.EVAL:
            return self.ev_task
        else:
            raise ValueError("Invalid stage")

    def start(self, stage: Stage, loader):
        super().start()
        task = self._get_task_from_stage(stage)
        self.update(task, total=len(loader))

    def update(self, stage: Stage, *args, **kwargs):
        task = self._get_task_from_stage(stage)
        self.update(task, *args, **kwargs)

    def stop(self, stage: Stage):
        task = self._get_task_from_stage(stage)
        self.stop(task)
