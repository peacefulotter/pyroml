# Inspired from: https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/callbacks/progress/rich_progress.py

from typing import TYPE_CHECKING, Optional

from rich import get_console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from typing_extensions import override

from pyroml.callbacks.progress.base_progress import BaseProgress
from pyroml.utils.log import get_logger

if TYPE_CHECKING:
    from pyroml.core.status import Status

log = get_logger(__name__)


class RichProgress(BaseProgress):
    def __init__(self):
        super().__init__()
        self.progress: Optional[Progress] = None

    @property
    def enabled(self):
        return self.progress is not None and super().enabled

    @override
    def setup(self):
        self._console = get_console()
        self._console.clear_live()
        self.progress = Progress(
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
        self.progress.start()

    @override
    def add_task(self, status: "Status", total: int, desc: str = None) -> int:
        task = self.progress.add_task(
            metrics="",
            total=total,
            description=desc,
        )
        return task

    @override
    def reset_task(self, task: int, total: int, desc: str = None):
        self.progress.reset(
            task,
            advance=0,
            total=total,
            visible=False,
            metrics="",
            description=desc,
        )

    @override
    def end_progress_task(self, task: int, hide: bool = False):
        self.progress.stop_task(task)
        if hide:
            self.progress.update(task, visible=False)

    @override
    def stop_progress(self):
        self.progress.stop()

    def _metrics_to_str(self, metrics: dict[str, float]):
        metrics_str = ""
        for name, value in metrics.items():
            if name == "stage":
                continue
            v = f"{value:.3f}" if isinstance(value, float) else value
            metrics_str += f"{name}={v} "
        return metrics_str

    @override
    def update_metrics(
        self,
        task: int,
        metrics: dict[str, float],
        advance: int = 1,
    ):
        metrics = self._metrics_to_str(metrics)
        self.progress.update(task, metrics=metrics, advance=advance)
        self.progress.refresh()
