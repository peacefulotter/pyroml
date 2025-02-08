from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyroml.utils.env import PyroEnv, get_bool_env
from pyroml.utils.log import get_logger

if TYPE_CHECKING:
    from pyroml.core.stage import Stage
    from pyroml.core.trainer import Trainer
    from pyroml.loop.base import Loop

log = get_logger(__name__)


@dataclass
class CallbackArgs:
    trainer: "Trainer"
    loop: "Loop"
    stage: "Stage"
    epoch: int
    step: int


class Callback:
    def _call_event(self, callback_name: str, args: CallbackArgs):
        fn = getattr(self, callback_name)
        if not callable(fn):
            return
        if get_bool_env(PyroEnv.VERBOSE):
            log.debug(
                f"Class {self.__class__.__name__} calling callback {callback_name}"
            )
        fn(args)

    # TODO: add stage independent callbacks?

    def on_train_start(self, args: CallbackArgs):
        pass

    def on_train_end(self, args: CallbackArgs):
        pass

    def on_train_iter_start(self, args: CallbackArgs):
        pass

    def on_train_iter_end(self, args: CallbackArgs):
        pass

    def on_train_epoch_start(self, args: CallbackArgs):
        pass

    def on_train_epoch_end(self, args: CallbackArgs):
        pass

    def on_validation_start(self, args: CallbackArgs):
        pass

    def on_validation_end(self, args: CallbackArgs):
        pass

    def on_validation_iter_start(self, args: CallbackArgs):
        pass

    def on_validation_iter_end(self, args: CallbackArgs):
        pass

    def on_validation_epoch_start(self, args: CallbackArgs):
        pass

    def on_validation_epoch_end(self, args: CallbackArgs):
        pass

    def on_predict_start(self, args: CallbackArgs):
        pass

    def on_predict_end(self, args: CallbackArgs):
        pass

    def on_predict_iter_start(self, args: CallbackArgs):
        pass

    def on_predict_iter_end(self, args: CallbackArgs):
        pass

    def on_predict_epoch_start(self, args: CallbackArgs):
        pass

    def on_predict_epoch_end(self, args: CallbackArgs):
        pass

    def on_exception(self, args: CallbackArgs):
        pass
