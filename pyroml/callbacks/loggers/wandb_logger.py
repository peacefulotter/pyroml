from typing import TYPE_CHECKING

try:
    import wandb

    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

from pyroml.callbacks.loggers.base_logger import BaseLogger
from pyroml.utils import get_classname

if TYPE_CHECKING:
    from pyroml.callbacks.callback import CallbackArgs
    from pyroml.core.model import PyroModule


# TODO: define generic LoggerCallback and integrate TensorboardLogger
# TODO: make CallbackLogger work for any stage
class WandBLogger(BaseLogger):
    def __init__(self, project_name: str):
        if not WANDB_AVAILABLE:
            raise "WandB is not installed but is needed for using WandbLogger, please: pip install wandb"
        super().__init__(project_name=project_name, env_key="WANDB_PROJECT")

    def _get_attr_names(self, model: "PyroModule"):
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

    def init(self, args: "CallbackArgs"):
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

    def log(self, args: "CallbackArgs", metrics: dict[str, float], on_epoch=True):
        status = args.status

        # TODO: add time
        # payload["time"] = time.time() - self.start_time
        # payload["dt_time"] = self.cur_time - old_time

        payload = {
            f"{status.stage.to_prefix()}/{'epoch-' if on_epoch else ''}{k}": v
            for k, v in metrics.items()
        }

        # Add status to payload
        status_dict = status.to_dict(json=True)
        status_dict.pop("stage")
        status_dict.pop("epoch")
        if on_epoch:
            status_dict.pop("step")
        payload.update(status_dict)

        if not on_epoch:
            payload.update(args.model.get_current_lr())

        print(payload)

        # payload = pd.json_normalize(payload, sep="/")
        # payload = payload.to_dict(orient="records")[0]

        wandb.log(payload)
