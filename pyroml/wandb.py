import time
import wandb
import logging
import pandas as pd

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler as Scheduler

from pyroml.utils import Stage
from pyroml.utils import get_lr
from pyroml.config import Config
from pyroml.model import PyroModel

log = logging.getLogger(__name__)


class Wandb:
    def __init__(self, config: Config):
        assert (
            config.wandb_project != None
        ), "When config.wandb is set, you need to specify a project name too (config.wandb_project='my_project_name')"
        self.config = config
        self.start_time = -1
        self.cur_time = -1

    def init(self, model: PyroModel, optimizer: Optimizer, scheduler: Scheduler):
        self.scheduler = scheduler

        run_name = self.get_run_name(optimizer, scheduler)

        wandb_config = self.config.__dict__
        classes_config = {
            "model": model.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
        }
        if scheduler != None:
            classes_config["scheduler"] = scheduler.__class__.__name__
        wandb_config.update(classes_config)

        log.info(
            f"Setting project_name {self.config.wandb_project} and run name {run_name}"
        )

        wandb.init(
            project=self.config.wandb_project,
            name=run_name,
            config=wandb_config,
        )
        wandb.define_metric("iter")
        wandb.define_metric("time")
        wandb.define_metric("eval", step_metric="iter")

    def log(self, stage: Stage, metrics: dict[str, float]):
        if self.start_time == -1:
            self.start_time = time.time()

        old_time = self.cur_time
        self.cur_time = time.time()

        payload = {f"{stage.value}/{k}": v for k, v in metrics.items()}
        payload["lr"] = get_lr(self.config, self.scheduler)
        payload["time"] = time.time() - self.start_time
        payload["dt_time"] = self.cur_time - old_time
        # payload = pd.json_normalize(payload, sep="/")
        # payload = payload.to_dict(orient="records")[0]

        wandb.log(payload)

    def get_run_name(self, optimizer, scheduler):
        optim_name = optimizer.__class__.__name__
        sched_name = scheduler.__class__.__name__ if scheduler != None else "None"
        name = f"{self.config.name}_lr={self.config.lr}_bs={self.config.batch_size}_optim={optim_name}_sched={sched_name}"
        return name
