import torch
from torch.utils.data import Dataset


class Prediction(torch.Tensor):
    """
    A torch.Tensor with a Dataset attribute
    """

    def __init__(self, other: torch.Tensor, dataset: int) -> None:
        pass

    def __new__(cls, other: torch.Tensor, dataset: Dataset, *args, **kwargs):
        instance = torch.Tensor._make_subclass(
            cls, other, require_grad=other.requires_grad
        )
        instance._dataset = dataset
        return instance

    @property
    def dataset(self) -> Dataset:
        return self._dataset


# b = torch.rand(2, 3).cuda()
# a = PyroPrediction(b, dataset=4)
# b = b.mul_(100)
# a.device, a.dataset
