import torch

# TODO: from sklearn.metrics import accuracy_score, precision_score, recall_score
# FIXME: move to device?


class Metric:
    def __init__(self, name, mode):
        self.name = name
        self.mode = mode
        self.value = 0
        self.mode_value = float("inf") if mode == "min" else float("-inf")

    @torch.no_grad()
    def compute(self, output, target):
        """
        Computes the metric for the given output and target tensors.

        Args:
            output (torch.Tensor): The output tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: A tensor containing a single value corresponding to the computed metric.
        """
        raise NotImplementedError

    @torch.no_grad()
    def aggregate(self, value):
        if isinstance(value, tuple) or isinstance(value, list):
            return torch.stack(value).mean()
        else:
            return value

    @torch.no_grad()
    def update(self, value):
        value = self.aggregate(value)
        self.value = value.item()
        compare_func = min if self.mode == "min" else max
        self.mode_value = compare_func(self.mode_value, self.value)
        return {
            self.name: self.value,
            f"{self.mode}_{self.name}": self.mode_value,
        }


class Accuracy(Metric):
    def __init__(self):
        super().__init__("acc", "max")

    @torch.no_grad()
    def compute(self, output, target):
        pred = output.argmax(dim=1, keepdim=True)
        correct = torch.sum(pred == target)
        return correct * 100.0 / output.shape[0]


class RMSE(Metric):
    def __init__(self):
        super().__init__("rmse", "min")
        self.size = 0

    @torch.no_grad()
    def compute(self, output, target):
        self.size += output.shape[0]
        return torch.sum((target - output) ** 2)

    @torch.no_grad()
    def aggregate(self, value):
        if isinstance(value, tuple) or isinstance(value, list):
            value = torch.stack(value)
        mse_mean = torch.sum(value) / self.size
        self.size = 0
        return torch.sqrt(mse_mean)


class Loss(Metric):
    def __init__(self, criterion):
        super().__init__("loss", "min")
        self.criterion = criterion

    @torch.no_grad()
    def compute(self, output, target):
        return self.criterion(output, target)
