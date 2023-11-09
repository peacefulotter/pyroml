import torch
from torch.utils.data import DataLoader

from .metrics import Loss
from .utils import to_device, get_lr


class Statistics:
    def __init__(self, model, criterion, scheduler, config, eval_dataset=None):
        assert (config.evaluate and eval_dataset != None) or (
            not config.evaluate
        ), "You have chosen to evaluate the model in the Config, but no evaluation dataset is passed"

        self.model = model
        self.criterion = criterion
        self.scheduler = scheduler
        self.config = config
        self.eval_loader = DataLoader(
            eval_dataset,
            shuffle=False,
            pin_memory=config.device != "cpu",
            batch_size=self.config.batch_size,
            num_workers=self.config.eval_num_workers,
        )
        self.lr = config.lr

        self.train_metrics = self.create_metrics()
        self.eval_metrics = self.create_metrics()

    def create_metrics(self):
        metrics = [Loss(self.criterion)]
        for Metric in self.config.metrics:
            metrics.append(Metric())
        return metrics

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        device = self.config.device
        metric_values = [[] for _ in range(len(self.eval_metrics))]

        for data, target in self.eval_loader:
            data, target = to_device(data, device), to_device(target, device)
            output = self.model(data).squeeze()  # FIXME: squeeze?

            for i, metric in enumerate(self.eval_metrics):
                metric_value = metric.compute(output, target)
                metric_values[i].append(metric_value)

        eval_stats = {}
        for metric, value in zip(self.eval_metrics, metric_values):
            stat = metric.update(value)
            eval_stats.update(stat)

        self.model.train()

        return eval_stats

    def log_stats(self, stats, epoch, iteration):
        log = f"[{epoch:03d} | {iteration:05d}:{self.config.max_iterations:05d}]"
        for metric in self.train_metrics:
            log += f" | [{metric.name}] tr: {stats['train'][metric.name]:.4f}"
            if "eval" in stats:
                log += f", ev: {stats['eval'][metric.name]:.4f}"
        log += f" | [Lr] {stats['lr']:.4f}"
        print(log)

    @torch.no_grad()
    def register(self, train_output, train_target, train_loss, epoch, iteration):
        # FIXME: move to device?

        train_stats = {}
        for metric in self.train_metrics:
            value = (
                train_loss
                if metric.name == "loss"
                else metric.compute(train_output, train_target)
            )
            stat = metric.update(value)
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
