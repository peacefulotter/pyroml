import torch
import torch.nn as nn

from enum import Enum
from typing import TypeAlias
from torchmetrics import Metric

from pyroml.utils import Stage


class Step(Enum):
    PRED = "pred"
    METRIC = "metric"
    TARGET = "target"


StepData: TypeAlias = torch.Tensor | dict[torch.Tensor] | tuple[torch.Tensor]
StepOutput: TypeAlias = dict[Step, torch.Tensor]


class PyroModel(nn.Module):

    def configure_metrics(
        self,
    ) -> dict[Metric] | None:
        pass

    def step(
        self,
        batch: StepData,
        stage: Stage,
    ) -> StepOutput:
        raise NotImplementedError("a step method must be implemented for your model")
