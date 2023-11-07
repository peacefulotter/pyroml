import wandb
import time


class Wandb:
    def __init__(self, model, optimizer, criterion, scheduler, config):
        assert (
            self.config.wandb_project != None
        ), "You need to specify a project name in the config to be able to use WandB (config.wandb_project='my_project_name')"
        self.config = config
        self.class_payload = {
            "model": model.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
            "criterion": criterion.__class__.__name__,
        }
        if scheduler != None:
            self.class_payload["scheduler"] = scheduler.__class__.__name__
        self.start = -1

    def init(self):
        run_name = self.get_run_name()
        if self.config.verbose:
            print(
                f"\t> Initializing wandb with project_name {self.config.wandb_project} and run name {run_name}"
            )
        wandb.init(
            project=self.config.wandb_project,
            name=run_name,
            config=self.config.__dict__,
        )
        wandb.define_metric("iter")
        wandb.define_metric("time")
        wandb.define_metric("eval", step_metric="iter")

    def log(self, stats):
        if self.start == -1:
            self.start = time.time()

        payload = dict(**stats, **self.class_payload)
        payload["lr"] = (
            self.config.lr
            if self.config.scheduler == None
            else self.config.scheduler.get_last_lr()[0]
        )
        payload["time"] = time.time() - self.start

        if self.config.verbose:
            print(f"\t> Logging to wandb: {payload}")

        wandb.log(payload)

    def get_run_name(self):
        name = f"{self.config.name}_lr={self.config.lr}_bs={self.config.batch_size}"
        return name
