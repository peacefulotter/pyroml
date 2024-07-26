from pyroml.utils import Stage
from pyroml.loop.eval import EvalLoop


class TestLoop(EvalLoop):
    @property
    def stage(self):
        return Stage.TEST
