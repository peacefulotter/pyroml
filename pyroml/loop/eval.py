import torch

from pyroml.core.stage import Stage
from pyroml.loop.base import Loop


class EvalLoop(Loop):
    @property
    def stage(self):
        return Stage.VAL

    @property
    def max_steps(self):
        return self.trainer.eval_max_steps

    @property
    def max_epochs(self):
        return 1

    @property
    def batch_size(self) -> int:
        return self.trainer.eval_batch_size

    @property
    def num_workers(self) -> int:
        return self.trainer.eval_num_workers

    def _run(self):
        self.model.eval()
        with torch.no_grad():
            return super()._run()
