import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torchmetrics.aggregation import MeanMetric
from torchmetrics import Metric, MetricTracker as _MetricTracker, MetricCollection

import pyroml as p
from pyroml.model import Step
from pyroml.callback import Callback

log = p.get_logger(__name__)

EPOCH_PREFIX = "epoch"


class LossMetric(MeanMetric):
    def __init__(self, loop: "p.Loop"):
        super().__init__()
        self.loss_fn = loop.trainer.loss

    def update(self, pred, target):
        loss = self.loss_fn(pred, target)
        super().update(loss)


class MissingStepKeyException(Exception):
    pass


class Metrics(MetricCollection):

    def __init__(self, loop: "p.Loop"):
        metrics = loop.model.configure_metrics()
        metrics = self._format_metrics(metrics)

        super().__init__(metrics)
        self.loss = LossMetric(loop)

    def _format_metrics(self, metrics: dict[Metric] | None) -> dict[str, Metric]:
        if metrics is None:
            return {}
        if isinstance(metrics, dict):
            return metrics
        raise ValueError(
            "The return type of the model.configure_metrics function should be a dict[torchmetrics.Metric], or None"
        )

    def update(self, pred, target):
        self.loss.update(pred, target)
        super().update(pred, target)

    def compute(self):
        loss_val = self.loss.compute()
        metric_vals = super().compute()
        return {**metric_vals, "loss": loss_val}


class MetricTracker(_MetricTracker, Callback):

    NO_NA_COLUMNS = ["stage", "epoch", "step"]

    def __init__(self, loop: "p.Loop"):
        super().__init__(Metrics(loop))

        self.status = loop.status

        self.records = pd.DataFrame([], columns=["stage", "epoch", "step"])

        self.current_step_metrics: dict[str, float] = {}
        self.current_epoch_metrics: dict[str, float] = {}

    def _detach(self, x):
        if isinstance(x, torch.Tensor):
            if len(x.shape) == 0:
                return x.item()
            else:
                return x.detach().cpu().numpy()
        return x

    def _extract_output(self, output: "p.StepOutput"):
        def __check(key: Step, metric_key: Step):
            out = None
            if metric_key in output:
                out = output[metric_key]
            elif metric_key not in output and key in output:
                msg = f"No metric in output, using {key} instead\nIf your model is used for classification, you likely want to output a {metric_key} key as well."
                # TODO: log this only once, on my machine it logs multiple times warnings.warn(msg, stacklevel=2)
                out = output[key]
            else:
                msg = f"No {metric_key} or {key} key in model.step output, your model should at least return a tensor associated with the {key} or {metric_key} key"
                raise MissingStepKeyException(msg)
            return out

        out_metric = __check(Step.PRED, Step.METRIC_PRED)
        out_target = __check(Step.TARGET, Step.METRIC_TARGET)

        return out_metric, out_target

    def _register_metrics(
        self,
        metrics: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        for col in metrics.keys():
            if col not in self.records.columns:
                self.records[col] = np.nan
        metrics = {k: self._detach(v) for k, v in metrics.items()}
        for k, v in self.status.to_dict().items():
            metrics[k] = v
        self.records.loc[len(self.records)] = metrics

    def forward(self, output: "p.StepOutput"):
        out_metric, out_target = self._extract_output(output)
        step_metrics = super().forward(out_metric, out_target)
        self._register_metrics(step_metrics)
        return step_metrics

    def _register_step_metrics(self, output: "p.StepOutput") -> dict[str, float]:
        step_metrics = self.forward(output)
        self.current_step_metrics = step_metrics
        return step_metrics

    # =================== epoch_start ===================

    def on_epoch_start(self):
        self.increment()

    def on_train_epoch_start(self, **kwargs: "p.CallbackKwargs"):
        self.on_epoch_start()

    def on_validation_epoch_start(self, **kwargs: "p.CallbackKwargs"):
        self.on_epoch_start()

    def on_test_epoch_start(self, **kwargs: "p.CallbackKwargs"):
        self.on_epoch_start()

    # =================== epoch_end ===================

    def _on_epoch_end(self, **kwargs: "p.CallbackKwargs"):
        # Since an EvalLoop is created inside a TrainLoop
        # we need to check if the current loop is the same as the one that created the tracker
        # otherwise the this method will be called twice
        loop = kwargs["loop"]
        if self.status != loop.status:
            return

        def prefix_cb(name: str):
            return f"{EPOCH_PREFIX}_{name}"

        epoch_metrics = self.compute()
        epoch_metrics = {prefix_cb(k): v for k, v in epoch_metrics.items()}
        self._register_metrics(epoch_metrics)
        self.current_epoch_metrics = epoch_metrics

    def on_train_epoch_end(self, **kwargs: "p.CallbackKwargs"):
        self._on_epoch_end(**kwargs)

    def on_validation_epoch_end(self, **kwargs: "p.CallbackKwargs"):
        self._on_epoch_end(**kwargs)

    def on_test_epoch_end(self, **kwargs: "p.CallbackKwargs"):
        self._on_epoch_end(**kwargs)

    # =================== api ===================

    def _records_filter_cols(self, with_epoch=True) -> pd.DataFrame:
        cols = [
            c
            for c in self.records.columns
            if (with_epoch and c.startswith(EPOCH_PREFIX))
            or (not with_epoch and not c.startswith(EPOCH_PREFIX))
        ]
        return self.records[cols].dropna()

    def get_step_records(self) -> pd.DataFrame:
        return self._records_filter_cols(with_epoch=False)

    def get_epoch_records(self) -> pd.DataFrame:
        return self._records_filter_cols(with_epoch=True)

    def get_last_step_metrics(self) -> dict[str, float]:
        return self.current_step_metrics

    def get_last_epoch_metrics(self) -> dict[str, float]:
        return self.current_epoch_metrics

    def step(self, output: "p.StepOutput") -> dict[str, float]:
        self._register_step_metrics(output)

    def plot(self, axs=None):
        raise NotImplementedError("This method is not implemented yet")
        ncols = len(self.metrics)
        _, axs = (
            (_, axs)
            if axs is not None
            else plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 5, 5))
        )
        for metric, ax in zip(self.metrics.values(), axs.flatten()):
            metric.plot(ax=ax)
        plt.tight_layout()
        plt.show()
