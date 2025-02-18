from pyroml.callbacks.loggers.base_logger import BaseLogger

try:
    from tensorboard import SummaryWriter

    _TENSORBOARD_AVAILABLE = True
except Exception:
    _TENSORBOARD_AVAILABLE = False


class TensorboardLogger(BaseLogger):
    def __init__(self, project_name, env_key):
        raise NotImplementedError("Tensorboard logger is not yet supported")
        if not _TENSORBOARD_AVAILABLE:
            raise "Tensorboard is not available, please install first using pip install tensorboard"
        super().__init__(project_name, env_key)

    @property
    def experiment(self):
        self._experiment = SummaryWriter(log_dir=self.log_dir, **self._kwargs)
