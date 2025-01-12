import torch
import warnings
from contextlib import AbstractContextManager, nullcontext

import pyroml as p

log = p.get_logger(__name__)


class Autocast(AbstractContextManager):
    def __init__(self, trainer: "p.Trainer"):
        device_type = trainer.device
        if device_type == "auto":
            device_type = "cuda" if torch.cuda.is_available() else "cpu"

        if device_type == "cpu" and trainer.dtype != torch.bfloat16:
            self.ctx = nullcontext()
            self.dtype = torch.float32
            msg = "Autocast is not supported on CPU with dtype != then bfloat16, data and model won't be casted automatically"
            log.warning(msg, stacklevel=2)
        else:
            self.ctx = torch.autocast(device_type=device_type, dtype=trainer.dtype)
            self.dtype = trainer.dtype

        self.device = torch.device(device=device_type)

        log.info(f"Using device {self.device}, dtype {self.dtype}")

    def __enter__(self):
        log.debug(f"Entering Autocast with device={self.device}, dtype={self.dtype}")
        return self.ctx

    def __exit__(self, exc_type, exc_value, traceback):
        log.debug(f"Exiting Autocast with device={self.device}, dtype={self.dtype}")
        return False
