import time
import wandb

import pyroml as p

from pyroml.callback import Callback
from pyroml.loop.status import Status
from pyroml.utils import Stage, get_classname


class Wandb(Callback):
    def __init__(self, loop: "p.Loop"):
        # TODO: merge wandb and wandb_project into single config.wandb
        assert (
            loop.trainer.wandb_project != None
        ), "When config.wandb is set, you need to specify a project name too (config.wandb_project='my_project_name')"
        self.loop = loop

        self.start_time = None
        self.cur_time = -1

    def on_train_start(
        self, trainer: "p.Trainer", loop: "p.Loop", **kwargs: "p.CallbackKwargs"
    ):
        self.start_time = -1
        self.cur_time = -1
        self._init()

    def on_train_iter_end(
        self, trainer: "p.Trainer", loop: "p.Loop", **kwargs: "p.CallbackKwargs"
    ):
        metrics = loop.tracker.get_last_step_metrics()
        self._log(metrics=metrics)

    def _on_end(self, loop: "p.Loop"):
        metrics = loop.tracker.get_last_epoch_metrics()
        self._log(metrics=metrics)

    def on_train_epoch_end(
        self, trainer: "p.Trainer", loop: "p.Loop", **kwargs: "p.CallbackKwargs"
    ):
        self._on_end(loop)

    def on_validation_end(
        self, trainer: "p.Trainer", loop: "p.Loop", **kwargs: "p.CallbackKwargs"
    ):
        self._on_end(loop)

    def _get_attr_names(self):
        m = self.loop.model
        attr_names = dict(
            model=get_classname(m),
            optim=get_classname(m.optimizer),
        )
        if hasattr(m, "scheduler") and m.scheduler != None:
            attr_names["sched"] = get_classname(m.scheduler)
        return attr_names

    def _init(self):
        run_name = self.get_run_name()

        # FIXME: make sure self.config is not modified by the .update()
        # TODO: also improve the .__dict__ usage; convert every value to string manually ?
        wandb_config = self.loop.trainer.__dict__
        attr_names = self._get_attr_names()
        wandb_config.update(attr_names)

        wandb.init(
            project=self.loop.trainer.wandb_project,
            name=run_name,
            config=wandb_config,
        )
        wandb.define_metric("epoch")
        wandb.define_metric("step")
        wandb.define_metric("time")
        # NOTE: is this necessary? : wandb.define_metric("eval", step_metric="iter")

    def _log(self, metrics: dict[str, float]):
        status = self.loop.status

        if self.start_time == -1:
            self.start_time = time.time()

        old_time = self.cur_time
        self.cur_time = time.time()

        print(metrics)
        payload = {f"{status.stage.to_prefix()}/{k}": v for k, v in metrics.items()}

        if status.stage == Stage.TRAIN:
            payload.update(status.to_dict())

        payload["lr"] = self.loop.model.get_current_lr()
        payload["time"] = time.time() - self.start_time
        payload["dt_time"] = self.cur_time - old_time
        # payload = pd.json_normalize(payload, sep="/")
        # payload = payload.to_dict(orient="records")[0]
        print(payload)

        wandb.log(payload)

    def get_run_name(self):
        attr_names = self._get_attr_names()

        run_name = "_".join(f"{attr}={name}" for attr, name in attr_names.items())
        run_name += f"_lr={self.loop.trainer.lr}_bs={self.loop.trainer.batch_size}"

        return run_name