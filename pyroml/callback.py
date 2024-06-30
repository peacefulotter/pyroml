from typing import TypedDict

import pyroml as p


class OnChangeKwargs(TypedDict):
    old_stage: "p.Stage"
    new_stage: "p.Stage"


class Callback:

    def on_stage_change(self, **kwargs: OnChangeKwargs):
        pass

    def on_train_start(self, trainer: "p.Trainer", epoch: int, step: int):
        pass

    def on_train_end(self, trainer: "p.Trainer", epoch: int, step: int):
        pass

    def on_train_iter_start(self, trainer: "p.Trainer", epoch: int, step: int):
        pass

    def on_train_iter_end(self, trainer: "p.Trainer", epoch: int, step: int, metrics):
        pass

    def on_train_epoch_start(self, trainer: "p.Trainer", epoch: int, step: int):
        pass

    def on_train_epoch_end(self, trainer: "p.Trainer", epoch: int, step: int):
        pass

    def on_validation_start(self, trainer: "p.Trainer", epoch: int, step: int):
        pass

    def on_validation_end(self, trainer: "p.Trainer", epoch: int, step: int):
        pass

    def on_validation_iter_start(self, trainer: "p.Trainer", epoch: int, step: int):
        pass

    def on_validation_iter_end(
        self, trainer: "p.Trainer", epoch: int, step: int, metrics
    ):
        pass

    def on_validation_epoch_start(self, trainer: "p.Trainer", epoch: int, step: int):
        pass

    def on_validation_epoch_end(self, trainer: "p.Trainer", epoch: int, step: int):
        pass

    def on_test_start(self, trainer: "p.Trainer", epoch: int, step: int):
        pass

    def on_test_end(self, trainer: "p.Trainer", epoch: int, step: int):
        pass

    def on_test_iter_start(self, trainer: "p.Trainer", epoch: int, step: int):
        pass

    def on_test_iter_end(self, trainer: "p.Trainer", epoch: int, step: int, metrics):
        pass

    def on_test_epoch_start(self, trainer: "p.Trainer", epoch: int, step: int):
        pass

    def on_test_epoch_end(self, trainer: "p.Trainer", epoch: int, step: int):
        pass
