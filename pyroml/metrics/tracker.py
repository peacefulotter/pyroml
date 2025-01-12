import torch
import warnings
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
    maximize = False
    higher_is_better = False

    def __init__(self, loss_fn: torch.nn.Module):
        super().__init__()
        self.loss_fn = loss_fn

    def update(self, pred, target):
        loss = self.loss_fn(pred, target)
        super().update(loss)


class MissingStepKeyException(Exception):
    pass


class PyroMetrics(Metric):
    def __init__(self, model: "p.PyroModel", loss_fn=torch.nn.Module):
        super().__init__()

        metrics = model.configure_metrics()
        metrics = self._format_metrics(metrics)

        self.metrics = MetricCollection(metrics)
        self.loss = LossMetric(loss_fn=loss_fn)
        self.maximize = [
            m.higher_is_better for m in [*self.metrics.values(), self.loss]
        ]

    def _format_metrics(self, metrics: dict[Metric] | None) -> dict[str, Metric]:
        if metrics is None:
            return {}
        if isinstance(metrics, dict):
            return metrics
        raise ValueError(
            "The return type of the model.configure_metrics function should be a dict[torchmetrics.Metric], or None"
        )

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        metric_pred: torch.Tensor,
        metric_target: torch.Tensor,
    ):
        self.loss.update(pred, target)
        self.metrics.update(metric_pred, metric_target)

    def compute(self):
        loss_val = self.loss.compute()
        metric_vals = self.metrics.compute()
        return {**metric_vals, "loss": loss_val}


class MetricTracker(_MetricTracker, Callback):
    def __init__(
        self, status: "p.Status", model: "p.PyroModel", loss_fn: torch.nn.Module
    ):
        super().__init__(PyroMetrics(model=model, loss_fn=loss_fn))
        self.global_status = status
        self.records = pd.DataFrame([], columns=["stage", "epoch", "step"])
        self.current_step_metrics: dict[str, float] = {}
        self.current_epoch_metrics: dict[str, float] = {}

    def _detach(self, x) -> float | np.ndarray:
        if isinstance(x, torch.Tensor):
            if len(x.shape) == 0:
                return x.item()
            else:
                return x.detach().cpu().numpy()
        return x

    def _extract_output(self, output: "p.StepOutput"):
        def get_val(key: Step):
            if key in output:
                return output[key]
            msg = f"No {key} key in model.step output, your model should return a tensor associated with the {key} key"
            raise MissingStepKeyException(msg)

        out_pred = get_val(Step.PRED)
        out_target = get_val(Step.TARGET)

        return out_pred, out_target

    def _extract_metrics_output(self, output: "p.StepOutput"):
        def get_val(key: Step, metric_key: Step):
            if metric_key in output:
                return output[metric_key]
            elif metric_key not in output and key in output:
                msg = f"No metric in output, using {key} instead\nIf your model is used for classification, you likely want to output a {metric_key} key as well."
                warnings.warn(msg)
                return output[key]
            msg = f"No {metric_key} or {key} key in model.step output, your model should at least return a tensor associated with the {key} or {metric_key} key"
            raise MissingStepKeyException(msg)

        out_pred = get_val(Step.PRED, Step.METRIC_PRED)
        out_target = get_val(Step.TARGET, Step.METRIC_TARGET)

        return out_pred, out_target

    def _register_metrics(
        self,
        metrics: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        for col in metrics.keys():
            if col not in self.records.columns:
                self.records[col] = np.nan
        metrics = {k: self._detach(v) for k, v in metrics.items()}
        for k, v in self.global_status.to_dict().items():
            metrics[k] = v
        self.records.loc[len(self.records)] = metrics

    def forward(self, output: "p.StepOutput"):
        pred, tgt = self._extract_output(output)
        metric_pred, metric_tgt = self._extract_metrics_output(output)
        step_metrics = super().forward(pred, tgt, metric_pred, metric_tgt)
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
        # otherwise this method will be called twice
        loop: "p.Loop" = kwargs["loop"]
        if self.global_status != loop.status:
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

    def _get_metrics_keys(self):
        m: PyroMetrics = self._base_metric
        return m.metrics.keys()

    def plot(
        self,
        stage: "p.Stage" = None,
        plot_keys: list[str] = None,
        epoch: bool = False,
        kind: str = "line",
    ):
        stage = stage or self.global_status.stage
        plot_keys = self._get_metrics_keys() if plot_keys is None else plot_keys
        prefix = EPOCH_PREFIX + "_" if epoch else ""
        plot_keys = [prefix + k for k in plot_keys]
        loss_key = prefix + "loss"
        x_key = "epoch" if epoch else "step"

        records = self.records
        records = records[records["stage"] == stage.value]

        fig, ax = plt.subplots()

        # For some plot kinds, we need to group everything otherwise metrics overlap the loss..
        # This is not ideal as it means loss and metrics share the y axis..
        if kind == "bar" or kind == "barh":
            records = records[[x_key, loss_key, *plot_keys]].dropna()
            records.plot(x=x_key, ax=ax, legend=False, kind=kind)

        # For other kinds, such as e.g. line, loss and metrics shouldn't share the same y axis
        else:
            # Plot the loss on the first axis
            ax.set_ylabel("Loss")
            loss_records = records[[x_key, loss_key]].dropna()
            loss_records.plot(x=x_key, ax=ax, legend=False, kind=kind)

            if len(plot_keys) > 0:
                # Create a separate axis for metrics plot
                ax2 = ax.twinx()
                ax2.set_ylabel("Metrics")

                # Second axis matches the first axis color cycle
                ax2._get_lines = ax._get_lines
                ax2._get_patches_for_fill = ax._get_patches_for_fill

                # Plot metrics on the secondary axis
                metrics_records = records[[x_key, *plot_keys]].dropna()
                metrics_records.plot(x=x_key, ax=ax2, legend=False, kind=kind)

        ax.set_title(stage.value)
        fig.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=5,
            fancybox=True,
            shadow=True,
        )

        plt.tight_layout()
        plt.show()
