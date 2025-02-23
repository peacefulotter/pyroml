from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from pyroml.callbacks.progress.tqdm_progress import TQDMProgress
from pyroml.core.trainer import Trainer

if TYPE_CHECKING:
    from pyroml.core import PyroModule


def freeze_module(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_last_layers(model: nn.Module, n: int, depth: int = 2, verbose=True):
    layers: dict[str, list[nn.Parameter]] = {}
    for name, param in model.named_parameters():
        key = "".join(name.split(".")[:depth])
        print(name, param.requires_grad)
        if key not in layers:
            layers[key] = []
        layers[key].append(param)
        param.requires_grad = False

    to_unfreeze = list(layers.keys())[-n:]

    for key in to_unfreeze:
        for param in layers[key]:
            param.requires_grad = True

    if verbose:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)


def num_params(model: "nn.Module", only_trainable=False):
    return sum(
        [p.numel() for p in model.parameters() if not only_trainable or p.requires_grad]
    )


def get_features(
    model: "PyroModule",
    dataset: Dataset,
    batch_size: int = 16,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "auto",
) -> torch.Tensor:
    temp_trainer = Trainer(
        lr=0,
        batch_size=batch_size,
        max_epochs=1,
        device=device,
        dtype=dtype,
        callbacks=[TQDMProgress()],
    )
    _, preds = temp_trainer.predict(model, dataset)
    return preds
