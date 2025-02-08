# From: https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/metrics.py


import torch


def _k_accuracy(correct: torch.Tensor, k: int, maxk: int, batch_size: int):
    return correct[: min(k, maxk)].reshape(-1).float().sum(0) * 100.0 / batch_size


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    if len(topk) == 1:
        return _k_accuracy(correct, topk[0], maxk, batch_size)
    return [_k_accuracy(correct, k, maxk, batch_size) for k in topk]


def binary_accuracy(output: torch.Tensor, target: torch.Tensor):
    output = output.squeeze()
    target = target.squeeze()
    return 100 * (output == target).sum() / output.numel()
