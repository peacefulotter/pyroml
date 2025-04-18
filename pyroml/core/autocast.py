from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING

import torch

from pyroml.utils.log import get_logger

if TYPE_CHECKING:
    pass


log = get_logger(__name__)


class Autocast(AbstractContextManager):
    def __init__(self, device: torch.device | str, dtype: torch.dtype) -> None:
        device_type = device
        if device_type == "auto":
            device_type = "cuda" if torch.cuda.is_available() else "cpu"

        if str(device_type) == "cpu" and dtype != torch.bfloat16:
            self.ctx = nullcontext()
            self.dtype = torch.float32
            msg = "Autocast is not supported on CPU with dtype != then bfloat16, data and model won't be casted automatically"
            log.warning(msg, stacklevel=2)
        else:
            self.ctx = torch.autocast(device_type=str(device_type), dtype=dtype)
            self.dtype = dtype

        self.device = torch.device(device=device_type)

        log.debug(f"Using device {self.device}, dtype {self.dtype}")

    def __enter__(self):
        log.debug(f"Entering Autocast with device={self.device}, dtype={self.dtype}")
        return self.ctx

    def __exit__(self, exc_type, exc_value, traceback):
        log.debug(f"Exiting Autocast with device={self.device}, dtype={self.dtype}")
        return False
