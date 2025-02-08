import torch.nn as nn


def unfreeze_last_layers(model: nn.Module, n: int, verbose=True):
    layers: dict[str, list[nn.Parameter]] = {}
    for name, param in model.named_parameters():
        key = "".join(name.split(".")[:2])
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
