from torch.utils.data import Dataset

from pyroml.utils import Stage
from pyroml.loop.eval import EvalLoop


class TestLoop(EvalLoop):

    def run(self, dataset: Dataset):
        self.model.to(self.autocast.device)
        super().run(dataset)
        self.model.cpu()

    @property
    def stage(self):
        return Stage.TEST
