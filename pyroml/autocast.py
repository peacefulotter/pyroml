import torch
import logging
from contextlib import AbstractContextManager, nullcontext

from pyroml.config import Config
from pyroml.callback import Callback, OnChangeKwargs

log = logging.getLogger(__name__)


class Autocast(AbstractContextManager, Callback):
    def __init__(self, config: Config):
        device_type = config.device
        if device_type == "auto":
            device_type = "cuda" if torch.cuda.is_available() else "cpu"

        if device_type == "cpu" and config.dtype != torch.bfloat16:
            self.ctx = nullcontext()
            self.dtype = torch.float32  # TODO: use dtype of model params?
        else:
            torch.autocast(device_type=device_type, dtype=config.dtype)
            self.dtype = config.dtype

        self.device = torch.device(device=device_type)

        log.info(f"Using device {self.device}, dtype {self.dtype}")

    def _on_start(self, **kwargs):
        model = kwargs["trainer"].model
        model.to(self.device)

    def _on_end(self, **kwargs):
        model = kwargs["trainer"].model
        model.cpu()

    def on_train_start(self, **kwargs):
        self._on_start(**kwargs)

    def on_test_start(self, **kwargs):
        self._on_start(**kwargs)

    def on_train_end(self, **kwargs):
        self._on_start(**kwargs)

    def on_test_end(self, **kwargs):
        self._on_start(**kwargs)

    def __enter__(self):
        log.debug(f"Entering Autocast with device={self.device}, dtype={self.dtype}")
        return self.ctx

    def __exit__(self, exc_type, exc_value, traceback):
        log.debug(f"Exiting Autocast with device={self.device}, dtype={self.dtype}")
        return False
