from torch.utils.data import Dataset

from pyroml.utils import Stage
from pyroml.loop.base import Loop


class EvalLoop(Loop):
    def run(self, dataset: Dataset):
        # Don't move to device cause it's assumed to be already there by the training loop
        # TODO: check if there is a cost of .to(device) when model already in device
        # If not the case, then move .to(device) / model.cpu() to autocast callbacks and add to loop callback array
        self.model.eval()
        super().run(dataset)

    @property
    def stage(self):
        return Stage.VAL

    @property
    def max_steps(self):
        return self.trainer.eval_max_steps

    @property
    def max_epochs(self):
        return 1
