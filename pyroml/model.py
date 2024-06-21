import torch
import torch.nn as nn

from typing import TypeAlias

from pyroml.utils import Stage
from pyroml.metrics import PyroMetric

StepData: TypeAlias = torch.Tensor | dict[torch.Tensor] | tuple[torch.Tensor]
StepOutput: TypeAlias = torch.Tensor | dict[torch.Tensor]


class PyroModel(nn.Module):

    def log(self, metric: PyroMetric, log: bool = True) -> None:
        # TODO: connect this to trainer and update metric + prog bar
        pass

    def step(
        self,
        batch: StepData,
        stage: Stage,
    ) -> StepOutput:
        raise NotImplementedError("a step function must be implemented for your model")
