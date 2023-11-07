import torch
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


class Record:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return f"Config({str(self.__dict__)[1:-1]})"

    def __repr__(self):
        return self.__str__()


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
