import os
import time
from typing import TYPE_CHECKING

import wandb

from pyroml.callbacks.callback import Callback
from pyroml.core.stage import Stage
from pyroml.utils import get_classname

if TYPE_CHECKING:
    from pyroml.callbacks.callback import CallbackArgs
    from pyroml.core.model import PyroModel


# TODO: define generic LoggerCallback and integrate TensorboardLogger
# TODO: make CallbackLogger work for any stage
class WandBLogger(Callback):
    def __init__(self, wandb_project: str):
        # TODO: merge wandb and wandb_project into single config.wandb
        assert wandb_project is not None, (
            "When using WandB logger, you need to specify a project name (wandb_project='my_project_name')"
        )
        self.wandb_project = self._get_wandb_project(wandb_project=wandb_project)
        self.reset_time()

    def _get_wandb_project(self, wandb_project: str | None):
        project = wandb_project or os.environ.get("WANDB_PROJECT")
        if project == "" or project is None:
            msg = "Wandb project name is required, please set WANDB_PROJECT in your environment variables or pass wandb_project in the WandBLogger constructor"
            raise ValueError(msg)
        return project

    def _get_attr_names(self, model: "PyroModel"):
        attr_names = dict(
            model=get_classname(model),
            optim=get_classname(model.optimizer),
        )
        if hasattr(model, "scheduler") and model.scheduler is not None:
            attr_names["sched"] = get_classname(model.scheduler)
        return attr_names

    def get_run_name(self, args: "CallbackArgs"):
        attr_names = self._get_attr_names(model=args.model)
        run_name = "_".join(f"{attr}={name}" for attr, name in attr_names.items())
        run_name += f"_lr={args.trainer.lr}_bs={args.trainer.batch_size}"
        return run_name

    def _init(self, args: "CallbackArgs"):
        run_name = self.get_run_name(args)

        # FIXME: make sure self.config is not modified by the .update()
        # TODO: also improve the .__dict__ usage; convert every value to string manually ?
        wandb_config = args.trainer.__dict__
        attr_names = self._get_attr_names(model=args.model)
        wandb_config.update(attr_names)

        wandb.init(
            project=self.wandb_project,
            name=run_name,
            config=wandb_config,
        )
        wandb.define_metric("epoch")
        wandb.define_metric("step")
        wandb.define_metric("time")
        # NOTE: is this necessary? : wandb.define_metric("eval", step_metric="iter")

    def reset_time(self):
        self.start_time = None
        self.cur_time = None

    # =================== on_start ===================

    def on_train_start(self, args: "CallbackArgs"):
        self.reset_time()
        self._init(args)

    def on_validation_start(self, args: "CallbackArgs"):
        self.reset_time()

    def on_predict_start(self, args: "CallbackArgs"):
        self.reset_time()

    # =================== iter_end ===================

    def on_train_iter_end(self, args: "CallbackArgs"):
        metrics = args.loop.tracker.get_last_step_metrics()
        self.log(args=args, metrics=metrics, on_epoch=False)

    # =================== epoch_end ===================

    def _on_epoch_end(self, args: "CallbackArgs"):
        metrics = args.loop.tracker.get_last_epoch_metrics()
        self.log(args=args, metrics=metrics)

    def on_train_epoch_end(self, args: "CallbackArgs"):
        self._on_epoch_end(args)

    def on_validation_end(self, args: "CallbackArgs"):
        self._on_epoch_end(args)

    def on_predict_end(self, args: "CallbackArgs"):
        self._on_epoch_end(args)

    # =================== api ===================

    def log(self, args: "CallbackArgs", metrics: dict[str, float], on_epoch=True):
        status = args.status

        if self.start_time is None:
            self.start_time = time.time()

        if self.cur_time is None:
            self.cur_time = time.time()

        old_time = self.cur_time
        self.cur_time = time.time()

        payload = {f"{status.stage.to_prefix()}/{k}": v for k, v in metrics.items()}

        if status.stage == Stage.TRAIN and not on_epoch:
            payload.update(status.to_dict(json=True))

        if not on_epoch:
            payload.update(args.model.get_current_lr())
            payload["time"] = time.time() - self.start_time
            payload["dt_time"] = self.cur_time - old_time

        print(payload)

        # payload = pd.json_normalize(payload, sep="/")
        # payload = payload.to_dict(orient="records")[0]

        wandb.log(payload)
