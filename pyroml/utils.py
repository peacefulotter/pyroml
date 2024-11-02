import os
import time
import torch
import random
import logging
import numpy as np
import torch.nn as nn

from enum import Enum

import pyroml as p

log = p.get_logger(__name__)


def get_classname(obj):
    return obj.__class__.__name__


def to_device(obj, device):
    if isinstance(obj, float) or isinstance(obj, int):
        return torch.tensor(obj).to(device)
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(to_device(v, device) for v in obj)
    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    return obj


def get_date():
    return time.strftime("%Y-%m-%d_%H:%M", time.gmtime(time.time()))


# https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L4788C1-L4799C21
def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class Stage(Enum):
    TRAIN = "train"
    VAL = "validation"
    TEST = "test"

    def to_prefix(self):
        return {
            Stage.TRAIN: "tr",
            Stage.VAL: "ev",
            Stage.TEST: "te",
        }[self]


def seed_everything(seed):
    log.info(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
