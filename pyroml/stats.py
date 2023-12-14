import torch
from torch.utils.data import DataLoader

from .metrics import Loss
from .utils import to_device, get_lr
from .logger import Logger


class Statistics:
    def __init__(self, model, criterion, scheduler, config, device, eval_dataset=None):
        assert (config.evaluate and eval_dataset != None) or (
            not config.evaluate
        ), "You have chosen to evaluate the model in the Config, but no evaluation dataset is passed"

        self.model = model
        self.criterion = criterion
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.eval_dataset = eval_dataset

        self.lr = config.lr

        self.train_metrics = self.create_metrics()
        self.eval_metrics = self.create_metrics()

        self.logger = Logger("Statistics", config)

    def create_metrics(self):
        metrics = [m for m in self.config.metrics]
        metrics.append(Loss(self.criterion))
        return metrics

    @torch.no_grad()
    def evaluate(self):
        self.logger.log("Evaluating")
        self.model.eval()

        metric_values = [[] for _ in range(len(self.eval_metrics))]

        self.eval_loader = DataLoader(
            self.eval_dataset,
            shuffle=False,
            pin_memory=self.device != "cpu",
            batch_size=self.config.eval_batch_size,
            num_workers=self.config.eval_num_workers,
        )

        for i, (data, target) in enumerate(self.eval_loader):
            if (
                self.config.max_eval_iterations != None
                and i >= self.config.max_eval_iterations
            ):
                break
            data, target = to_device(data, self.device), to_device(target, self.device)
            output = self.model(data)

            for i, metric in enumerate(self.eval_metrics):
                metric_value = metric.compute(output, target, evaluate=True)
                metric_values[i].append(metric_value)

        eval_stats = {}
        for metric, value in zip(self.eval_metrics, metric_values):
            stat = metric.update(value, evaluate=True)
            eval_stats.update(stat)

        self.model.train()

        return eval_stats

    def log_stats(self, stats, epoch, iteration):
        log = f"[epoch] {epoch:03d} | [iter] {iteration:05d}:{self.config.max_iterations:05d}"
        for metric in self.train_metrics:
            log += f" | [{metric.name}] tr: {stats['train'][metric.name]:.4f}"
            if "eval" in stats:
                log += f", ev: {stats['eval'][metric.name]:.4f}"
        log += f" | [lr] {stats['lr']:.4f}"
        self.logger.log(log)

    @torch.no_grad()
    def register(self, train_output, train_target, train_loss, epoch, iteration):
        train_stats = {}
        for metric in self.train_metrics:
            value = (
                train_loss
                if metric.name == "loss"
                else metric.compute(train_output, train_target, evaluate=False)
            )
            stat = metric.update(value, evaluate=False)
            train_stats.update(stat)

        self.lr = get_lr(self.config, self.scheduler)

        stats = {"epoch": epoch, "iter": iteration, "train": train_stats, "lr": self.lr}

        if self.config.evaluate != False:
            eval_epoch = (
                self.config.evaluate == "epoch"
                and epoch % self.config.evaluate_every == 0
            )
            eval_iter = (
                self.config.evaluate == True
                and iteration % self.config.evaluate_every == 0
            )
            if eval_epoch or eval_iter:
                eval_stats = self.evaluate()
                stats["eval"] = eval_stats

        if self.config.verbose:
            self.log_stats(stats, epoch, iteration)

        return stats
