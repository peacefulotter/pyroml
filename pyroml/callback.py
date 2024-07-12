from typing import TypedDict

import pyroml as p


class CallbackKwargs(TypedDict):
    epoch: int
    step: int


class MetricsKwargs(CallbackKwargs):
    metrics: dict[str, float]


class Callback:

    # TODO: remove trainer from arg?
    # TODO: move everything to typed kwargs?

    def on_train_start(
        self, trainer: "p.Trainer", loop: "p.Loop", **kwargs: CallbackKwargs
    ):
        pass

    def on_train_end(
        self, trainer: "p.Trainer", loop: "p.Loop", **kwargs: CallbackKwargs
    ):
        pass

    def on_train_iter_start(
        self, trainer: "p.Trainer", loop: "p.Loop", **kwargs: CallbackKwargs
    ):
        pass

    def on_train_iter_end(
        self, trainer: "p.Trainer", loop: "p.Loop", **kwargs: MetricsKwargs
    ):
        pass

    def on_train_epoch_start(
        self, trainer: "p.Trainer", loop: "p.Loop", **kwargs: CallbackKwargs
    ):
        pass

    def on_train_epoch_end(
        self, trainer: "p.Trainer", loop: "p.Loop", **kwargs: CallbackKwargs
    ):
        pass

    def on_validation_start(
        self, trainer: "p.Trainer", loop: "p.Loop", **kwargs: CallbackKwargs
    ):
        pass

    def on_validation_end(
        self, trainer: "p.Trainer", loop: "p.Loop", **kwargs: CallbackKwargs
    ):
        pass

    def on_validation_iter_start(
        self, trainer: "p.Trainer", loop: "p.Loop", **kwargs: CallbackKwargs
    ):
        pass

    def on_validation_iter_end(
        self, trainer: "p.Trainer", loop: "p.Loop", **kwargs: MetricsKwargs
    ):
        pass

    def on_validation_epoch_start(
        self, trainer: "p.Trainer", loop: "p.Loop", **kwargs: CallbackKwargs
    ):
        pass

    def on_validation_epoch_end(
        self, trainer: "p.Trainer", loop: "p.Loop", **kwargs: CallbackKwargs
    ):
        pass

    def on_test_start(
        self, trainer: "p.Trainer", loop: "p.Loop", **kwargs: CallbackKwargs
    ):
        pass

    def on_test_end(
        self, trainer: "p.Trainer", loop: "p.Loop", **kwargs: CallbackKwargs
    ):
        pass

    def on_test_iter_start(
        self, trainer: "p.Trainer", loop: "p.Loop", **kwargs: CallbackKwargs
    ):
        pass

    def on_test_iter_end(
        self, trainer: "p.Trainer", loop: "p.Loop", **kwargs: MetricsKwargs
    ):
        pass

    def on_test_epoch_start(
        self, trainer: "p.Trainer", loop: "p.Loop", **kwargs: CallbackKwargs
    ):
        pass

    def on_test_epoch_end(
        self, trainer: "p.Trainer", loop: "p.Loop", **kwargs: CallbackKwargs
    ):
        pass
