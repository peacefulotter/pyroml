import torch
import logging
import warnings
import numpy as np
import pandas as pd

from torchmetrics import Metric

import pyroml as p
from pyroml.model import PyroModel, Step
from pyroml.callback import Callback

from .loss import LossMetric

log = logging.getLogger(__name__)


class MissingStepKeyException(Exception):
    pass


class MetricsTracker(Callback):

    NO_NA_COLUMNS = ["stage", "epoch", "step"]

    def __init__(self, loop: "p.Loop"):
        self.model = loop.model
        self.status = loop.status

        self.metrics = self.model.configure_metrics()
        self.metrics: dict[str, Metric] = self._format_metrics(self.metrics)
        self.metrics["loss"] = LossMetric(loop=loop)

        self.records = pd.DataFrame([], columns=["stage", "epoch", "step"])

        self.current_step_metrics: dict[str, float] = {}
        self.current_epoch_metrics: dict[str, float] = {}

    def _format_metrics(self, metrics: dict[Metric] | None):
        if metrics is None:
            return {}
        if isinstance(metrics, dict):
            return metrics
        raise ValueError(
            "The return type of the model.configure_metrics function should be a dict[torchmetrics.Metric], or None"
        )

    def _detach(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
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
        metric_cb,
        prefix_cb=None,
    ) -> dict[str, float]:

        metrics = {}
        record = self.records

        for name, metric in self.metrics.items():
            if prefix_cb is not None:
                name = prefix_cb(name)

            if name not in record.columns:
                record[name] = np.nan

            metrics[name] = metric_cb(metric).item()

        record_metrics = dict(**self.status.to_dict(), **metrics)
        record.loc[len(record)] = record_metrics

        return metrics

    def _register_step_metrics(self, output: "p.StepOutput") -> dict[str, float]:
        out_metric, out_target = self._extract_output(output)

        def metric_cb(m: Metric):
            if isinstance(m, LossMetric):
                return m(out_metric, out_target, output=output)
            return m(out_metric, out_target)

        step_metrics = self._register_metrics(metric_cb=metric_cb)
        return step_metrics

    # NOTE: Maybe for later if we do trainer.predict() / trainer.test()
    # def compute(
    #     self, model: PyroModel, stage: Stage, output: StepOutput
    # ):

    def on_train_epoch_end(
        self, trainer: "p.Trainer", loop: "p.Loop", **kwargs: "p.CallbackKwargs"
    ):
        print("on_train_epoch_end")

        def prefix_cb(name: str):
            return f"epoch_{name}"

        def metric_cb(m: Metric):
            return m.compute()

        self.current_epoch_metrics = self._register_metrics(
            prefix_cb=prefix_cb, metric_cb=metric_cb
        )

    def _records_filter_cols(self, with_epoch=True) -> pd.DataFrame:
        cols = [
            c
            for c in self.records.columns
            if (with_epoch and "epoch" in c) or (not with_epoch and "epoch" not in c)
        ]

        return self.records[cols].dropna()

    def get_step_records(self) -> pd.DataFrame:
        return self._records_filter_cols(with_epoch=False)

    def get_epoch_records(self) -> pd.DataFrame:
        return self._records_filter_cols(with_epoch=True)

    def get_last_step_metrics(self):
        return self.current_step_metrics

    def get_last_epoch_metrics(self):
        return self.current_epoch_metrics

    def update(self, output: "p.StepOutput") -> dict[str, float]:
        self.current_step_metrics = self._register_step_metrics(output)

    def plot(self):
        # TODO: subplots + pass ax=ax to metric.plot()
        for _, metric in self.metrics.items():
            metric.plot()
