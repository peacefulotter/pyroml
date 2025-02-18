import os
from typing import Optional, TYPE_CHECKING
from pyroml.callbacks.callback import Callback
from pyroml.utils.classes import get_classname


if TYPE_CHECKING:
    from pyroml.callbacks import CallbackArgs


class BaseLogger(Callback):
    def __init__(self, project_name: Optional[str], env_key: Optional[str]):
        self.project_name = self._get_project_name(
            project_name=project_name, env_key=env_key
        )

    def _get_project_name(self, project_name: Optional[str], env_key: Optional[str]):
        project = project_name or (
            os.environ.get(env_key) if env_key is not None else None
        )
        if project == "" or project is None:
            msg = f"Project name is required for logger, please pass a project_name to {get_classname(self)}"
            if env_key is not None:
                msg += f" or set {env_key} in your environment variables"
            raise ValueError(msg)
        return project

    # =================== api ===================

    def init(self, args: "CallbackArgs"):
        raise NotImplementedError()

    def log(self, args: "CallbackArgs", metrics: dict[str, float], on_epoch: bool):
        raise NotImplementedError()

    # =================== on_start ===================

    def on_train_start(self, args: "CallbackArgs"):
        self.init(args)

    def on_validation_start(self, args: "CallbackArgs"):
        pass

    def on_predict_start(self, args: "CallbackArgs"):
        pass

    # =================== iter_end ===================

    def on_train_iter_end(self, args: "CallbackArgs"):
        metrics = args.loop.tracker.get_last_step_metrics()
        self.log(args=args, metrics=metrics, on_epoch=False)

    # =================== epoch_end ===================

    def _on_epoch_end(self, args: "CallbackArgs"):
        metrics = args.loop.tracker.get_last_epoch_metrics()
        self.log(args=args, metrics=metrics, on_epoch=True)

    def on_train_epoch_end(self, args: "CallbackArgs"):
        self._on_epoch_end(args)

    def on_validation_end(self, args: "CallbackArgs"):
        self._on_epoch_end(args)

    def on_predict_end(self, args: "CallbackArgs"):
        self._on_epoch_end(args)
