import time
import wandb

import pyroml as p

from pyroml.status import Status
from pyroml.config import Config
from pyroml.model import PyroModel
from pyroml.callback import Callback
from pyroml.utils import Stage, __classname


class Wandb(Callback):
    def __init__(self, model: PyroModel, config: Config):
        assert (
            config.wandb_project != None
        ), "When config.wandb is set, you need to specify a project name too (config.wandb_project='my_project_name')"
        self.model = model
        self.config = config

        self.start_time = None
        self.cur_time = -1

    def on_train_start(self, **kwargs):
        self.start_time = -1
        self.cur_time = -1
        self._init(self.model)

    def on_train_iter_end(self, trainer: "p.Trainer"):
        metrics = trainer.metrics_tracker.update()
        self._log(status=trainer.status, **metrics)

    def _on_end(self, trainer: "p.Trainer"):
        metrics = trainer.metrics_tracker.get_epoch_metrics()
        self._log(status=trainer.status, **metrics)

    def on_train_end(self, trainer: "p.Trainer", **kwargs):
        self._on_end(trainer)

    def on_validation_end(self, trainer: "p.Trainer", **kwargs):
        self._on_end(trainer)

    def _get_attr_names(self):
        attr_names = dict(
            model=__classname(self.model),
            optim=__classname(self.model.optimizer),
        )
        if self.model.scheduler != None:
            attr_names["sched"] = __classname(self.model.scheduler)
        return attr_names

    def _init(self):
        run_name = self.get_run_name()

        # FIXME: make sure self.config is not modified by the .update()
        # TODO: also make improve the .__dict__ usage; convert every value to string
        wandb_config = self.config.__dict__
        attr_names = self._get_attr_names()
        wandb_config.update(attr_names)

        wandb.init(
            project=self.config.wandb_project,
            name=run_name,
            config=wandb_config,
        )
        wandb.define_metric("epoch")
        wandb.define_metric("step")
        wandb.define_metric("time")
        # NOTE: what was this doing: wandb.define_metric("eval", step_metric="iter")

    def _log(self, status: Status, metrics: dict[str, float]):
        if self.start_time == -1:
            self.start_time = time.time()

        old_time = self.cur_time
        self.cur_time = time.time()

        payload = {f"{status.stage.value}/{k}": v for k, v in metrics.items()}

        if status.stage == Stage.TRAIN:
            payload.update(status.to_dict())

        payload["lr"] = self.model.get_current_lr()
        payload["time"] = time.time() - self.start_time
        payload["dt_time"] = self.cur_time - old_time
        # payload = pd.json_normalize(payload, sep="/")
        # payload = payload.to_dict(orient="records")[0]

        wandb.log(payload)

    def get_run_name(self):
        attr_names = self._get_attr_names()

        run_name = "_".join(f"{attr}={name}" for attr, name in attr_names.items())
        run_name += f"_lr={self.config.lr}_bs={self.config.batch_size}"

        return run_name
