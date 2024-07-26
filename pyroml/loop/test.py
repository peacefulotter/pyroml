from torch.utils.data import Dataset

import pyroml as p
from pyroml.utils import Stage
from pyroml.loop.eval import EvalLoop
from pyroml.loop.progress_bar import ProgressBar


class TestLoop(EvalLoop):

    # def __init__(self, trainer: "p.Trainer", model: "p.PyroModel") -> None:
    #     super().__init__(trainer, model)

    @property
    def stage(self):
        return Stage.TEST

    def run(self, dataset: Dataset):
        with self.progress.bar:
            super().run(dataset)
