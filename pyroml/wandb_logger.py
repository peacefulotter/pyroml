import time
import wandb
import logging

from pyroml.config import Config
from pyroml.model import PyroModel
from pyroml.utils import Stage, Callback, __classname

log = logging.getLogger(__name__)


class Wandb:
    def __init__(self, model: PyroModel, config: Config):
        assert (
            config.wandb_project != None
        ), "When config.wandb is set, you need to specify a project name too (config.wandb_project='my_project_name')"
        self.model = model
        self.config = config

        self.start_time = None
        self.cur_time = -1

        self.trainer.add_callback(Callback.ON_START(Stage.TRAIN), self.on_train_start)
        self.trainer.add_callback(
            Callback.ON_ITER_END(Stage.TRAIN), self.on_train_iter_end
        )
        self.trainer.add_callback(Callback.ON_END(Stage.TRAIN), self.on_train_end)
        self.trainer.add_callback(Callback.ON_END(Stage.VAL), self.on_validation_end)

    def on_train_start(self, **kwargs):
        self.start_time = -1
        self.cur_time = -1

        self.init(self.model)

    def on_train_iter_end(self, **kwargs):
        metrics = self.trainer.tracker.update(Stage.TRAIN)
        self.log(stage=Stage.TRAIN, **metrics)

    def on_train_end(self):
        metrics = self.trainer.tracker.get_epoch_metrics(Stage.TRAIN)
        self.log(stage=Stage.TRAIN, **metrics)

    def on_validation_end(self):
        metrics = self.trainer.tracker.get_epoch_metrics(Stage.VAL)
        self.log(stage=Stage.VAL, **metrics)

    def _get_attr_names(self):
        attr_names = dict(
            model=__classname(self.model),
            optim=__classname(self.model.optimizer),
        )
        if self.model.scheduler != None:
            attr_names["sched"] = __classname(self.model.scheduler)
        return attr_names

    def init(self):
        run_name = self.get_run_name()

        # FIXME: make sure self.config is not modified by the .update()
        # TODO: also make improve the .__dict__ usage; convert every value to string
        wandb_config = self.config.__dict__
        attr_names = self._get_attr_names()
        wandb_config.update(attr_names)

        log.info(
            f"Setting project_name {self.config.wandb_project} and run name {run_name}"
        )

        wandb.init(
            project=self.config.wandb_project,
            name=run_name,
            config=wandb_config,
        )
        # NOTE: following is necessary?
        wandb.define_metric("iter")
        wandb.define_metric("time")
        # NOTE: what was this doing: wandb.define_metric("eval", step_metric="iter")

    def log(self, stage: Stage, metrics: dict[str, float], epoch: int, step: int):
        if self.start_time == -1:
            self.start_time = time.time()

        old_time = self.cur_time
        self.cur_time = time.time()

        payload = {f"{stage.value}/{k}": v for k, v in metrics.items()}
        if stage == Stage.TRAIN:
            payload["epoch"] = epoch
            payload["step"] = step
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
