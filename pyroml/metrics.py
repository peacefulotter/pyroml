import torch
from torchmetrics import Metric


class PyroMetric(Metric):
    def __init__(self, name, best_func, keep_history=False):
        self.name = name
        self.best_func = best_func
        self.keep_history = keep_history

        self.value = torch.nan
        self.best_value = torch.nan

        self.history_value = []
        self.history_best_value = []

    def update(self, value, **kwargs):
        self.best_value = self.best_func(value, self.value)
        self.value = value

        self.history_value.append(value)
        self.history_best_value.append(self.best_value)


# TODO: use metric.__dict__ for logs / wandb?
# TODO: what to do for metrics that are not scalars, especially ones that have multiple fields such as e.g. (fn, ft, tn, tp)?
