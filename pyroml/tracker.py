import torch
import numpy as np
import pandas as pd

from pyroml.utils import Stage
from pyroml.model import StepOutput


class Tracker:
    def __init__(self):
        self.stats = pd.DataFrame([], columns=["epoch"])
        self.key_step = {}
        self.step = 0

    def _detach(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            return self._detach(x)
        elif isinstance(x, np.ndarray) and x.size == 1:
            return x.item()
        return x

    def _prefix(self, stage: Stage, k: str):
        if stage == Stage.TRAIN:
            return k
        return f"{stage.to_prefix()}_{k}"

    def _unroll_output(self, stage: Stage, output: StepOutput):
        if isinstance(output, dict):
            return {self._prefix(stage, k): self._detach(v) for k, v in output.items()}
        return self._unroll_output(stage, {"loss": output})

    def update(
        self, stage: Stage, epoch: int, output: StepOutput
    ) -> dict[str, float | np.ndarray]:
        output = self._unroll_output(stage, output)

        last_metrics = {}
        for k, v in self.key_step.items():
            last_metrics[k] = self.stats.at[v, k]

        step_metrics = {"epoch": epoch}
        for k, v in output.items():
            if k not in self.stats.columns:
                self.stats[k] = np.nan
            step_metrics[k] = v
            self.key_step[k] = self.step

        self.stats.loc[len(self.stats)] = step_metrics
        self.step += 1
        return last_metrics
