import wandb
import time
import pandas as pd

from .utils import get_lr


class Wandb:
    def __init__(self, config):
        assert (
            config.wandb_project != None
        ), "You need to specify a project name in the config to be able to use WandB (config.wandb_project='my_project_name')"
        self.config = config
        self.start = -1

    def init(self, model, optimizer, criterion, scheduler):
        self.scheduler = scheduler

        run_name = self.get_run_name(optimizer, scheduler)

        wandb_config = self.config.__dict__
        classes_config = {
            "model": model.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
            "criterion": criterion.__class__.__name__,
        }
        if scheduler != None:
            classes_config["scheduler"] = scheduler.__class__.__name__
        wandb_config.update(classes_config)

        if self.config.verbose:
            print(
                f"[WandB] Initializing wandb with project_name {self.config.wandb_project} and run name {run_name}"
            )

        wandb.init(
            project=self.config.wandb_project,
            name=run_name,
            config=wandb_config,
        )
        wandb.define_metric("iter")
        wandb.define_metric("time")
        wandb.define_metric("eval", step_metric="iter")

    def log(self, stats):
        if self.start == -1:
            self.start = time.time()

        payload = dict(**stats)
        payload["lr"] = get_lr(self.config, self.scheduler)
        payload["time"] = time.time() - self.start
        payload = pd.json_normalize(payload, sep="/")
        payload = payload.to_dict(orient="records")[0]

        wandb.log(payload)

    def get_run_name(self, optimizer, scheduler):
        optim_name = optimizer.__class__.__name__
        sched_name = scheduler.__class__.__name__ if scheduler != None else "None"
        name = f"{self.config.name}_lr={self.config.lr}_bs={self.config.batch_size}_optim={optim_name}_sched={sched_name}"
        return name
