from pyroml.utils import Stage


class State:
    def __init__(self, stage: Stage):
        self.stage = stage
        self.epoch = 0
        self.step = 0
