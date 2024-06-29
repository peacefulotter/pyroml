import torch
import logging
import warnings
import numpy as np
import pandas as pd

from torchmetrics import Metric

from pyroml.utils import Stage
from pyroml.model import PyroModel, StepOutput, Step

log = logging.getLogger(__name__)


class MissingStepKeyException(Exception):
    pass


class MetricsTracker:
    def __init__(self, model: PyroModel):
        self.model = model

        self.records: dict[Stage, pd.DataFrame] = {}
        self.metrics: dict[Stage, dict[str, Metric]] = {}

    def _format_metrics(self, metrics: dict[Metric] | None):
        if metrics is None:
            return {}
        if isinstance(metrics, dict):
            return metrics
        raise ValueError(
            "The return type of the model.configure_metrics function should be a dict[torchmetrics.Metric], or None"
        )

    def _init_stage_metrics(self, stage: Stage):
        metrics = self.model.configure_metrics()
        metrics = self._format_metrics(metrics)
        self.metrics[stage] = metrics
        self.records[stage] = pd.DataFrame([], columns=["epoch", "step"])

    def _detach(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        return x

    def _extract_output(self, output: StepOutput):
        if Step.TARGET not in output:
            msg = f"No target in output, your model should return a target tensor associated to the {Step.TARGET} key"
            raise MissingStepKeyException(msg)
        out_target = output[Step.TARGET]

        out_metric = None
        if Step.METRIC in output:
            out_metric = output[Step.METRIC]
        elif Step.METRIC not in output and Step.PRED in output:
            msg = f"No metric in output, using {Step.PRED} instead\nIf your model is used for classification, you likely want to use the {Step.METRIC} key."
            # TODO: log this only once, on my machine it logs multiple times warnings.warn(msg, stacklevel=2)
            out_metric = output[Step.PRED]
        else:
            msg = f"No {Step.METRIC} or {Step.PRED} key output, your model should at least return a tensor associated with the {Step.PRED} key"
            raise MissingStepKeyException(msg)

        return out_metric, out_target

    def _register_metrics(
        self,
        stage: Stage,
        epoch: int,
        step: int,
        metric_cb,
        prefix_cb=None,
    ):

        metrics = {}
        record = self.records[stage]

        for name, metric in self.metrics[stage].items():
            if prefix_cb is not None:
                name = prefix_cb(name)

            if name not in record.columns:
                record[name] = np.nan

            metrics[name] = metric_cb(metric).item()

        record_metrics = dict(epoch=epoch, step=step, **metrics)
        record.loc[len(record)] = record_metrics

        return metrics

    def _register_batch_metrics(
        self, stage: Stage, output: StepOutput, epoch: int, step: int
    ):
        out_metric, out_target = self._extract_output(output)

        batch_metrics = self._register_metrics(
            stage, epoch, step, metric_cb=lambda m: m(out_metric, out_target)
        )

        # FIXME: Find a better way of integrating the loss into the metrics here (that will propagate to wandb and progress bar)
        # What if we add the config.loss to the metrics dict in the tracker?
        if "loss" in output:
            batch_metrics["loss"] = output["loss"]
            if "loss" not in self.records[stage].columns:
                self.records[stage]["loss"] = np.nan
            N = len(self.records[stage]) - 1
            self.records[stage].loc[N, "loss"] = output["loss"]

        return batch_metrics

    def _register_epoch_metrics(self, stage: Stage, epoch: int, step: int):
        epoch_metrics = self._register_metrics(
            stage, epoch, step, metric_cb=lambda m: m.compute()
        )

        return epoch_metrics

    # NOTE: Maybe for later if we do trainer.predict() / trainer.test()
    # def compute(
    #     self, model: PyroModel, stage: Stage, output: StepOutput
    # ):

    def get_epoch_metrics(self, stage: Stage):
        records = self.records[stage]
        records_epoch = records.loc[:, records.columns.str.contains("epoch")]
        return records_epoch.to_dict(orient="records")

    def update(
        self, stage: Stage, output: StepOutput, epoch: int, step: int
    ) -> dict[str, float]:
        if stage not in self.metrics:
            self._init_stage_metrics(stage)

        # Register batch metrics
        metrics = self._register_batch_metrics(stage, output, epoch, step)

        # Register epoch metrics
        if epoch != self.records[stage].iloc[-1].epoch:
            epoch_metrics = self._register_epoch_metrics(stage, epoch, step)
            metrics.update(epoch_metrics)

        return metrics

    def plot(self, stage: Stage):
        # TODO: subplots + pass ax=ax to metric.plot()
        for _, metric in self.metrics[stage].items():
            metric.plot()
