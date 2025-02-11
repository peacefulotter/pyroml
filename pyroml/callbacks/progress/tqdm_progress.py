# Inspired from: https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/callbacks/progress/tqdm_progress.py

import importlib
import importlib.util
from typing import TYPE_CHECKING

from typing_extensions import override

from pyroml.callbacks.progress.base_progress import BaseProgress
from pyroml.core.stage import Stage
from pyroml.utils.log import get_logger

if importlib.util.find_spec("ipywidgets") is not None:
    from tqdm.auto import tqdm as _tqdm
else:
    from tqdm import tqdm as _tqdm

if TYPE_CHECKING:
    from tqdm import tqdm as TqdmType

    from pyroml.core.status import Status

log = get_logger(__name__)

BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}{postfix}]"


class Tqdm(_tqdm):
    pass


class TQDMProgress(BaseProgress):
    def __init__(self, stack_bars: bool = True):
        super().__init__(stack_bars=stack_bars, rich_colors=False)
        self.bars: dict[int, "TqdmType"] = {}

    @override
    def setup(self):
        self.bars = {}

    @override
    def add_task(self, status: "Status", total: int, desc: str = None) -> int:
        task = len(self.bars)
        bar: "TqdmType" = Tqdm(
            desc=desc,
            leave=self.stack_bars or status.stage == Stage.TRAIN,
            total=total,
            position=task,
            dynamic_ncols=True,
            bar_format=BAR_FORMAT,
        )
        self.bars[task] = bar
        return task

    @override
    def reset_task(self, task: int, total: int, desc: str = None):
        bar = self.bars[task]
        bar.reset(total)
        bar.initial = 0
        bar.set_description(desc)

    @override
    def end_progress_task(self, task: int, hide: bool = False):
        bar = self.bars[task]
        # bar.leave = hide
        bar.close()
        if bar.leave or hide:
            del self.bars[task]

    @override
    def stop_progress(self):
        pass

    @override
    def update_metrics(
        self,
        task: int,
        metrics: dict[str, float],
        advance: int = 1,
    ):
        bar = self.bars[task]
        bar.set_postfix(metrics)
        bar.update(n=advance)
