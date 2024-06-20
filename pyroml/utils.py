import time
import torch
import torch.nn as nn

from enum import Enum
from collections import defaultdict


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


def get_lr(config, scheduler):
    if scheduler == None:
        return config.lr
    return float(scheduler.get_last_lr()[0])


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


class Callbacks:
    def __init__(self):
        self.callbacks = defaultdict(list)

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str, **kwargs):
        for callback in self.callbacks.get(onevent, []):
            callback(self, **kwargs)


class Stage(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    def to_progress(self):
        return {
            Stage.TRAIN: "Training",
            Stage.VAL: "Validation",
            Stage.TEST: "Testing",
        }[self]
