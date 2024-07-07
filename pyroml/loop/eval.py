from torch.utils.data import Dataset

from pyroml.utils import Stage
from pyroml.loop.base import Loop


class EvalLoop(Loop):
    def run(self, dataset: Dataset):
        # Don't move to device cause it's assumed to be already there by the training loop
        self.model.eval()
        super().run(dataset)

    @property
    def stage(self):
        return Stage.VAL

    @property
    def max_steps(self):
        raise self.trainer.eval_max_steps

    @property
    def max_epochs(self):
        raise 1
