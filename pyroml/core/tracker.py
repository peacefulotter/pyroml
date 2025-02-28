import warnings
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from typing_extensions import override

from pyroml.callbacks import Callback
from pyroml.core.stage import Stage
from pyroml.utils.log import get_logger

if TYPE_CHECKING:
    from pyroml.callbacks.callback import CallbackArgs

log = get_logger(__name__)

EPOCH_PREFIX = "epoch"


def get_epoch_columns(df: pd.DataFrame):
    return list(filter(lambda c: c.startswith(EPOCH_PREFIX), df.columns))


def with_epoch_prefix(name: str):
    return f"{EPOCH_PREFIX}_{name}"


def detach(x: float | np.ndarray | torch.Tensor) -> float | np.ndarray:
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.item()
        else:
            return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray) and x.size == 1:
        return x.item()
    return x


class MetricsTracker(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.records = pd.DataFrame(
            {
                "stage": pd.Series([], dtype="str"),
                "epoch": pd.Series([], dtype="int32"),
                "step": pd.Series([], dtype="Int64"),
            }
        )
        self._current_step_metrics: dict[str, float | np.ndarray | torch.Tensor] = {}
        self._current_epoch_metrics: dict[str, float | np.ndarray] = {}

    def _register_metrics(
        self, metrics: dict[str, float | np.ndarray], args: "CallbackArgs"
    ):
        for col in metrics.keys():
            if col not in self.records.columns:
                self.records[col] = np.nan

        metrics["stage"] = args.loop.stage.value
        metrics["epoch"] = args.loop.epoch

        self.records.loc[len(self.records)] = metrics

    def log(self, **data: float | np.ndarray | torch.Tensor):
        data = {k: detach(v) for k, v in data.items()}
        self._current_step_metrics.update(data)

    # =================== epoch_start ===================

    def _on_epoch_start(self, args: "CallbackArgs"):
        self._current_epoch_metrics = {}

    @override
    def on_train_epoch_start(self, args: "CallbackArgs"):
        self._on_epoch_start(args)

    @override
    def on_validation_epoch_start(self, args: "CallbackArgs"):
        self._on_epoch_start(args)

    @override
    def on_predict_epoch_start(self, args: "CallbackArgs"):
        self._on_epoch_start(args)

    # =================== iter_start ===================
    def _on_iter_start(self):
        self._current_step_metrics = dict()

    @override
    def on_train_iter_start(self, args: "CallbackArgs"):
        self._on_iter_start()

    @override
    def on_validation_iter_start(self, args: "CallbackArgs"):
        self._on_iter_start()

    @override
    def on_predict_iter_start(self, args: "CallbackArgs"):
        self._on_iter_start()

    # =================== epoch_end ===================

    def _on_epoch_end(self, args: "CallbackArgs"):
        """
        Compute epoch metrics as the average over the current epoch
        """
        metrics = self.records[
            (self.records["stage"] == args.loop.stage.value)
            & (self.records["epoch"] == args.loop.epoch)
        ].copy()

        metrics.drop(
            columns=["stage", "step", "epoch", *get_epoch_columns(metrics)],
            inplace=True,
        )

        # Early exit if empty metrics dataframe
        if len(metrics) == 0:
            return

        metrics = metrics.mean().to_dict()
        self._current_epoch_metrics.update(metrics)

        # And register them in the records dataframe
        metrics["step"] = -1
        self._register_metrics(metrics, args)

    @override
    def on_train_epoch_end(self, args: "CallbackArgs"):
        self._on_epoch_end(args)

    @override
    def on_validation_epoch_end(self, args: "CallbackArgs"):
        self._on_epoch_end(args)

    @override
    def on_predict_epoch_end(self, args: "CallbackArgs"):
        self._on_epoch_end(args)

    # =================== iter_end ===================

    def _on_iter_end(self, args: "CallbackArgs") -> dict[str, float]:
        metrics = self._current_step_metrics
        metrics = {k: detach(v) for k, v in metrics.items()}
        metrics["step"] = args.loop.step
        self._register_metrics(metrics, args)
        return metrics

    @override
    def on_train_iter_end(self, args: "CallbackArgs"):
        self._on_iter_end(args)

    @override
    def on_validation_iter_end(self, args: "CallbackArgs"):
        self._on_iter_end(args)

    @override
    def on_predict_iter_end(self, args: "CallbackArgs"):
        self._on_iter_end(args)

    # =================== api ===================

    def get_step_records(self) -> pd.DataFrame:
        return self.records[self.records["step"] > 0]

    def get_epoch_records(self) -> pd.DataFrame:
        return self.records[self.records["step"] < 0]

    def get_last_step_metrics(self) -> dict[str, float]:
        return self._current_step_metrics

    def get_last_epoch_metrics(self) -> dict[str, float]:
        return self._current_epoch_metrics

    def plot(
        self,
        stage: Optional["Stage"] = None,
        plot_keys: Optional[list[str]] = None,
        epoch: bool = False,
        kind: str = "line",
        # grouped: bool = True,
    ):
        # assert grouped or (kind != "bar" and kind != "barh"), (
        #     f"grouped=False is incompatible with kind={kind}"
        # )

        x_key = "epoch" if epoch else "step"
        stages = (
            [stage.value]
            if stage is not None
            else self.records["stage"].unique().tolist()
        )

        plot_keys = sorted(
            set(plot_keys or self.records.columns) - set(["stage", "step", "epoch"])
        )
        records = self.get_epoch_records() if epoch else self.get_step_records()

        if len(records) == 0:
            warnings.warn("Attempting to plot tracker records with 0 entries")
            return

        # Since step counter is reset every epoch, we prevent plotting the step metrics over each other
        # By overriding the step column
        if not epoch:
            for s in records["stage"].unique():
                mask = records["stage"] == s
                records.loc[mask, "step"] = np.arange(len(records[mask]))

        nrows = len(stages)
        ncols = len(plot_keys)
        figsize = (ncols * 3, nrows * 3)
        fig = plt.figure(figsize=figsize, constrained_layout=True)

        subfigs = fig.subfigures(nrows=nrows, ncols=1, squeeze=False).flatten()
        for subfig, _stage in zip(subfigs, stages):
            stage_records = records[records["stage"] == _stage]

            subfig.suptitle(_stage.capitalize(), fontweight="bold")
            axs = subfig.subplots(nrows=1, ncols=ncols, squeeze=False)
            prev_ax = axs[0]
            for _ax, k in zip(axs, plot_keys):
                title = with_epoch_prefix(k) if epoch else k
                _ax.set_title(title)
                _ax._get_lines = prev_ax._get_lines
                _ax._get_patches_for_fill = prev_ax._get_patches_for_fill

                r = stage_records[[x_key, k]]  # .dropna()
                r.plot(x=x_key, ax=_ax, legend=False, kind=kind)

                prev_ax = _ax

        # axes: np.ndarray
        # fig, axes = plt.subplots(
        #     nrows=nrows,
        #     ncols=ncols,
        #     figsize=figsize,
        #     layout="constrained",
        #     squeeze=False,
        # )

        # if single_plot:
        #     from matplotlib.axes import Axes

        #     ax: Axes = ax.item()
        #     ax.set_title(stage.value)

        #     # For some plot kinds, we need to group everything otherwise metrics overlap the loss..
        #     # This is not ideal as it means loss and metrics share the y axis..
        #     if grouped:
        #         r = records[[x_key, *plot_keys]].dropna()
        #         r.plot(x=x_key, ax=ax, legend=False, kind=kind)

        #     # For other kinds, such as e.g. line, loss and metrics shouldn't share the same y axis
        #     else:
        #         for k in plot_keys:
        #             # Plot metrics
        #             r = records[[x_key, k]].dropna()
        #             r.plot(x=x_key, ax=ax, legend=False, kind=kind)

        #             # Create a separate axis
        #             new_ax = ax.twinx()
        #             new_ax.set_ylabel("Metrics")

        #             # New axis matches the previous axis color cycle
        #             new_ax._get_lines = ax._get_lines
        #             new_ax._get_patches_for_fill = ax._get_patches_for_fill

        #             ax = new_ax

        fig.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.1),
            ncol=len(plot_keys),
            fancybox=True,
            shadow=True,
        )

        # plt.tight_layout()
        plt.show()
