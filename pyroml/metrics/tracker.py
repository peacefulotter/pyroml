import torch
import logging
import warnings
import numpy as np
import pandas as pd

from torchmetrics import Metric

import pyroml as p
from pyroml.model import PyroModel, Step

log = logging.getLogger(__name__)


class MissingStepKeyException(Exception):
    pass


class MetricsTracker:
    def __init__(self, loop: "p.Loop"):
        self.model = loop.model
        self.status = loop.status

        self.metrics = self.model.configure_metrics()
        self.metrics: dict[str, Metric] = self._format_metrics(self.metrics)

        self.records = pd.DataFrame([], columns=["epoch", "step"])

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

    def _register_batch_metrics(self, output: "p.StepOutput") -> dict[str, float]:
        out_metric, out_target = self._extract_output(output)

        def metric_cb(m: Metric):
            return m(out_metric, out_target)

        batch_metrics = self._register_metrics(metric_cb=metric_cb)

        # FIXME: Find a better way of integrating the loss into the metrics here (that will propagate to wandb and progress bar)
        # What if we add the config.loss to the metrics dict in the tracker?
        if "loss" in output:
            record = self.records
            batch_metrics["loss"] = output["loss"]
            if "loss" not in record.columns:
                record["loss"] = np.nan
            N = len(record) - 1
            record.loc[N, "loss"] = output["loss"]

        return batch_metrics

    def _register_epoch_metrics(self):
        def metric_cb(m: Metric):
            return m.compute()

        epoch_metrics = self._register_metrics(metric_cb=metric_cb)
        return epoch_metrics

    # NOTE: Maybe for later if we do trainer.predict() / trainer.test()
    # def compute(
    #     self, model: PyroModel, stage: Stage, output: StepOutput
    # ):

    # TODO: refactor the following get metrics methods
    # maybe save epoch metrics in different attribute
    # on get metrics, merge based on step?

    def get_batch_metrics(self):
        records_batch = self.records.loc[
            -1:, ~self.records.columns.str.contains("epoch")
        ]
        return records_batch.to_dict(orient="list")

    def get_epoch_metrics(self):
        records_epoch = self.records.loc[
            -1:, self.records.columns.str.contains("epoch")
        ]
        return records_epoch.to_dict(orient="list")

    def update(self, output: "p.StepOutput") -> dict[str, float]:
        stage, epoch = self.status.stage, self.status.epoch

        # Register batch metrics
        metrics = self._register_batch_metrics(output)

        # Register epoch metrics
        # TODO: on callback? on_epoch_end
        if epoch != self.records.iloc[-1].epoch:
            epoch_metrics = self._register_epoch_metrics()
            metrics.update(epoch_metrics)

        return metrics

    def plot(self):
        # TODO: subplots + pass ax=ax to metric.plot()
        for _, metric in self.metrics.items():
            metric.plot()
