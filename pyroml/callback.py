import os
import logging
from typing import TypedDict

import pyroml as p
from .env import get_bool_env, PyroEnv


class CallbackKwargs(TypedDict):
    trainer: "p.Trainer"
    loop: "p.Loop"
    stage: "p.Stage"
    epoch: int
    step: int


class MetricsKwargs(CallbackKwargs):
    metrics: dict[str, float]


log = p.get_logger(__name__)


class Callback:
    def __getattribute__(self, attr: str):
        methods = [
            func
            for func in dir(Callback)
            if callable(getattr(Callback, func)) and func.startswith("on_")
        ]
        if attr in methods and get_bool_env(PyroEnv.VERBOSE):
            log.debug(f"Class {self.__class__.__name__} calling callback {attr}")

        return super().__getattribute__(attr)

    def on_train_start(self, **kwargs: CallbackKwargs):
        pass

    def on_train_end(self, **kwargs: CallbackKwargs):
        pass

    def on_train_iter_start(self, **kwargs: CallbackKwargs):
        pass

    def on_train_iter_end(self, **kwargs: MetricsKwargs):
        pass

    def on_train_epoch_start(self, **kwargs: CallbackKwargs):
        pass

    def on_train_epoch_end(self, **kwargs: CallbackKwargs):
        pass

    def on_validation_start(self, **kwargs: CallbackKwargs):
        pass

    def on_validation_end(self, **kwargs: CallbackKwargs):
        pass

    def on_validation_iter_start(self, **kwargs: CallbackKwargs):
        pass

    def on_validation_iter_end(self, **kwargs: MetricsKwargs):
        pass

    def on_validation_epoch_start(self, **kwargs: CallbackKwargs):
        pass

    def on_validation_epoch_end(self, **kwargs: CallbackKwargs):
        pass

    def on_test_start(self, **kwargs: CallbackKwargs):
        pass

    def on_test_end(self, **kwargs: CallbackKwargs):
        pass

    def on_test_iter_start(self, **kwargs: CallbackKwargs):
        pass

    def on_test_iter_end(self, **kwargs: MetricsKwargs):
        pass

    def on_test_epoch_start(self, **kwargs: CallbackKwargs):
        pass

    def on_test_epoch_end(self, **kwargs: CallbackKwargs):
        pass
