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
    def __init__(self, model: PyroModel):
        self.model = model

        self.records: dict["p.Stage", pd.DataFrame] = {}
        self.metrics: dict["p.Stage", dict[str, Metric]] = {}

    def _format_metrics(self, metrics: dict[Metric] | None):
        if metrics is None:
            return {}
        if isinstance(metrics, dict):
            return metrics
        raise ValueError(
            "The return type of the model.configure_metrics function should be a dict[torchmetrics.Metric], or None"
        )

    def _init_stage_metrics(self, stage: "p.Stage"):
        metrics = self.model.configure_metrics()
        metrics = self._format_metrics(metrics)
        self.metrics[stage] = metrics
        self.records[stage] = pd.DataFrame([], columns=["epoch", "step"])

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
        stage: "p.Stage",
        metric_cb,
        prefix_cb=None,
    ):

        metrics = {}
        record = self.records[stage]

        for name, metric in self.metric.items():
            if prefix_cb is not None:
                name = prefix_cb(name)

            if name not in record.columns:
                record[name] = np.nan

            metrics[name] = metric_cb(metric).item()

        record_metrics = dict(**self.status.to_dict(), **metrics)
        record.loc[len(record)] = record_metrics

        return metrics

    def _register_batch_metrics(self, stage: "p.Stage", output: "p.StepOutput"):
        out_metric, out_target = self._extract_output(output)

        batch_metrics = self._register_metrics(
            stage, metric_cb=lambda m: m(out_metric, out_target)
        )

        # FIXME: Find a better way of integrating the loss into the metrics here (that will propagate to wandb and progress bar)
        # What if we add the config.loss to the metrics dict in the tracker?
        if "loss" in output:
            record = self.records[stage]
            batch_metrics["loss"] = output["loss"]
            if "loss" not in record.columns:
                record["loss"] = np.nan
            N = len(record) - 1
            record.loc[N, "loss"] = output["loss"]

        return batch_metrics

    def _register_epoch_metrics(self, stage: "p.Stage"):
        epoch_metrics = self._register_metrics(stage, metric_cb=lambda m: m.compute())
        return epoch_metrics

    # NOTE: Maybe for later if we do trainer.predict() / trainer.test()
    # def compute(
    #     self, model: PyroModel, stage: Stage, output: StepOutput
    # ):

    # TODO: refactor the following get metrics methods

    def get_batch_metrics(self, stage: "p.Stage"):
        record = self.records[stage]
        records_batch = record.loc[-1:, ~record.columns.str.contains("epoch")]
        return records_batch.to_dict(orient="list")

    def get_epoch_metrics(self, stage: "p.Stage"):
        record = self.records[stage]
        records_epoch = record.loc[-1:, record.columns.str.contains("epoch")]
        return records_epoch.to_dict(orient="list")

    def update(
        self, stage: "p.Stage", output: "p.StepOutput", epoch: int
    ) -> dict[str, float]:
        if stage not in self.metrics:
            self._init_stage_metrics()

        # Register batch metrics
        metrics = self._register_batch_metrics(output)

        # Register epoch metrics
        if epoch != self.records[stage].iloc[-1].epoch:
            epoch_metrics = self._register_epoch_metrics()
            metrics.update(epoch_metrics)

        return metrics

    def plot(self, stage: "p.Stage" = None):
        # TODO: subplots + pass ax=ax to metric.plot()
        stages = [stage] if stage is not None else self.metrics.keys()
        for stage in stages:
            for _, metric in self.metrics[stage].items():
                metric.plot()
