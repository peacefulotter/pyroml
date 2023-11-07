import torch
from torch.utils.data import DataLoader

from .wandb import Wandb
from .utils import to_device

# TODO: from sklearn.metrics import accuracy_score, precision_score, recall_score


class Statistics:
    def __init__(
        self, model, optimizer, criterion, scheduler, config, eval_dataset=None
    ):
        assert (config.evaluate and eval_dataset != None) or (
            not config.evaluate
        ), "You have chosen to evaluate the model in the Config, but no evaluation dataset is passed"

        self.train = {
            "loss": 0,
            "min_loss": 1e9,
            "acc": 0,
            "max_acc": 0,
            "rmse": 0,
            "min_rmse": 1e9,
        }
        self.eval = {
            "loss": 0,
            "min_loss": 1e9,
            "acc": 0,
            "max_acc": 0,
            "rmse": 0,
            "min_rmse": 1e9,
        }
        self.model = model
        self.criterion = criterion
        self.scheduler = scheduler
        self.config = config
        self.eval_dataset = eval_dataset
        self.eval_loader = DataLoader(
            eval_dataset,
            shuffle=False,
            pin_memory=False,  # FIXME: set this to true
            batch_size=self.config.batch_size,
            num_workers=0,
        )
        self.lr = config.lr
        self.wandb = Wandb(model, optimizer, criterion, scheduler, config)

    @torch.no_grad()
    def get_accuracy(self, output, target):
        pred = output.argmax(dim=1, keepdim=True)
        correct = torch.sum(pred == target)
        return correct * 100.0 / output.shape[0]

    @torch.no_grad()
    def get_rmse(self, output, target):
        return self._get_rmse(self.get_mse(output, target))

    @torch.no_grad()
    def _get_rmse(self, mse):
        return torch.sqrt(mse)

    @torch.no_grad()
    def get_mse(self, output, target):
        return torch.sum((target - output) ** 2)

    def _register_stats(self, obj, acc, loss, rmse):
        obj["acc"] = acc.item()
        obj["max_acc"] = max(obj["max_acc"], obj["acc"])

        obj["loss"] = loss.item()
        obj["min_loss"] = min(obj["min_loss"], obj["loss"])

        obj["rmse"] = rmse.item()
        obj["min_rmse"] = min(obj["min_rmse"], obj["rmse"])

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        device = self.config.device
        losses, accuracies, mses = [], [], []

        for data, target in self.eval_loader:
            data, target = to_device(data, device), to_device(target, device)
            output = self.model(data).squeeze()

            loss = self.criterion(output, target)
            losses.append(loss)

            accuracy = self.get_accuracy(output, target)
            accuracies.append(accuracy)

            mse = self.get_mse(output, target)
            mses.append(mse)

        eval_loss = torch.stack(to_device(losses, device)).mean()
        eval_accuracy = torch.stack(to_device(accuracies, device)).mean()
        eval_mse = torch.sum(torch.stack(to_device(mses, device))) / len(
            self.eval_dataset
        )
        eval_rmse = self._get_rmse(eval_mse)

        self.model.train()

        return eval_loss, eval_accuracy, eval_rmse

    @torch.no_grad()
    def register(self, train_output, train_target, train_loss, epoch, iteration):
        if self.config.stats_every == None or (
            self.config.stats_every != 0 and iteration % self.config.stats_every != 0
        ):
            return
        # device = self.config.device
        # FIXME: move to device?

        train_acc = self.get_accuracy(train_output, train_target)
        train_rmse = self.get_rmse(train_output, train_target)
        self._register_stats(self.train, train_acc, train_loss, train_rmse)

        self.lr = self.scheduler.get_lr() if self.scheduler else self.lr

        stats = {
            "epoch": epoch,
            "iter": iteration,
            "train": self.train,
        }

        if self.config.evaluate != False:
            eval_epoch = (
                self.config.evalute == "epoch"
                and epoch % self.config.evaluate_every == 0
            )
            eval_iter = (
                self.config.evalute == True
                and iteration % self.config.evaluate_every == 0
            )
            if eval_epoch or eval_iter:
                eval_loss, eval_accuracy, eval_rmse = self.evaluate()
                self._register_stats(self.eval, eval_accuracy, eval_loss, eval_rmse)
                stats["eval"] = self.eval

        if self.config.wandb:
            self.wandb.log(stats)
        return stats
